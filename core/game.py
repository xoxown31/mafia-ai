from typing import List, Dict, Tuple, Optional
import config
import random
import json

# Import RationalCharacter directly
from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent


class MafiaGame:
    def __init__(self, log_file=None):
        # players 리스트에는 RationalCharacter가 들어갑니다.
        self.players: List[LLMAgent] = []
        self.phase = config.PHASE_DAY_DISCUSSION  # Start with discussion phase
        self.day_count = 1
        self.alive_status = []
        self.vote_counts = []
        self.log_file = log_file
        self.final_vote = 0
        self.last_execution_result = None  # Track execution results
        self.last_night_result = None  # Track night results
        self.police_logs = {}  # 경찰 조사 결과 기록

        # 역할 번호 -> 역할 이름 문자열 맵
        self.role_names = {
            v: k.replace("ROLE_", "")
            for k, v in config.__dict__.items()
            if k.startswith("ROLE_")
        }

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> Dict:
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = config.PHASE_DAY_DISCUSSION  # Start with discussion phase
        self.players = []
        self.last_execution_result = None
        self.last_night_result = None

        # 1. 플레이어 생성 (All use LLMAgent)
        for i in range(config.PLAYER_COUNT):
            # Create rational agent for all players
            player = LLMAgent(player_id=i)
            self.players.append(player)

        # 2. 역할 할당
        roles_to_assign = config.ROLES.copy()
        random.shuffle(roles_to_assign)

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int

            # 로그 출력
            role_name = self.role_names.get(role_int, "Unknown")
            self._log(f"플레이어 {i}: {role_name} (LLM Agent)")

        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]

    def process_turn(
        self, action: int, ai_claim_role: int = -1
    ) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        # Merged phases: PHASE_DAY_CLAIM removed
        if self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion()
            self.phase = config.PHASE_DAY_VOTE
        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote()
            self.phase = config.PHASE_DAY_EXECUTE
        elif self.phase == config.PHASE_DAY_EXECUTE:
            self._process_day_execute()
            self.phase = config.PHASE_NIGHT
        elif self.phase == config.PHASE_NIGHT:
            self._process_night()
            self.phase = config.PHASE_DAY_DISCUSSION
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def _process_day_discussion(self):
        self._log("  - 낮 토론 시작")
        structured_claims = []

        MAX_DEBATE_ROUNDS = 2
        for debate_round in range(MAX_DEBATE_ROUNDS):
            for p in self.players:
                if not p.alive:
                    continue

                game_state_dict = self._get_game_status(p.id)
                # 대화 기록을 넘길 때 화자 ID를 포함한 JSON 전달
                conversation_log_str = json.dumps(structured_claims, ensure_ascii=False)

                response_str = p.get_action(game_state_dict, conversation_log_str)
                try:
                    claim_dict = json.loads(response_str)
                except:
                    continue

                if claim_dict.get("discussion_status") == "End":
                    break

                claim = claim_dict.get("claim")
                role_id = claim_dict.get("role")
                target_id = claim_dict.get("target_id")
                role_name = self.role_names.get(role_id, "알 수 없음")
                discussion_ended = claim_dict.get("discussion_status")
                reason = claim_dict.get("reason", "")

                # 발언 로깅
                if claim == 0:  # 본인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {role_name}이라 주장합니다."
                elif claim == 1:  # 타인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {target_id}는 {role_name}라고 주장합니다!"
                else:
                    speech = f"- 플레이어 {p.id}이(가) 침묵합니다."

                self._log(speech)
                self._log(f"  - 플레이어 {p.id}의 주장 이유: {reason}")
                # structured_claims에 추가
                structured_claims.append(
                    {
                        "speech": speech,
                        "claim": claim,
                    }
                )

            if discussion_ended == "End":
                break

    def _process_day_vote(self):
        player_count = len(self.players)
        self.vote_counts = [0] * player_count
        structured_votes = []

        # 투표 기록 초기화
        for p in self.players:
            p.voted_by_last_turn = []
        self._log("  - 낮 투표: 플레이어들이 처형 대상을 선택합니다.")

        # 봇들 투표
        for p in self.players:
            if not p.alive:
                continue
            game_state_dict = self._get_game_status(p.id)
            conversation_log_str = json.dumps(structured_votes, ensure_ascii=False)

            # get_action 호출 및 JSON 파싱
            response_str = p.get_action(game_state_dict, conversation_log_str)
            try:
                vote_dict = json.loads(response_str)
            except json.JSONDecodeError:
                self._log(
                    f"  - 플레이어 {p.id}: 잘못된 JSON 응답 수신. 투표를 건너뜁니다."
                )
                continue

            target = vote_dict.get("target_id")
            speech = vote_dict.get("reason")
            # [FIX] target이 None인 경우를 방지하여 TypeError를 막음
            if target is not None and target != -1:
                self.vote_counts[target] += 1
                self._log(f"- 플레이어 {p.id}이(가) {target}에게 투표했습니다.")
                self._log(f"  이유: {speech}")

                self.players[target].voted_by_last_turn.append(p.id)
                self.players[target].vote_history[p.id] += 1
            else:
                self._log(f"- 플레이어 {p.id}이(가) 투표를 기권했습니다.")

            structured_votes.append(
                {
                    "target_id": target,
                    "speech": speech,
                }
            )

        self._log(f"  - 최종 투표 집계: {self.vote_counts}")

    def _process_day_execute(self):
        # 처형 집행
        max_votes = max(self.vote_counts)
        self.final_vote = 0
        structured_executes = []

        if max_votes > 0:
            executed_targets = [
                i for i, v in enumerate(self.vote_counts) if v == max_votes
            ]

            if len(executed_targets) > 1:
                self._log(
                    f"  - 투표 동률 발생: {executed_targets}에 처형이 무산되었습니다."
                )
                self.last_execution_result = None
            else:
                executed_target = executed_targets[0]
                for p in self.players:
                    if not p.alive:
                        continue
                    game_state_dict = self._get_game_status(p.id)
                    conversation_log_str = json.dumps(
                        structured_executes, ensure_ascii=False
                    )
                    # get_action 호출 및 JSON 파싱
                    response_str = p.get_action(game_state_dict, conversation_log_str)
                    try:
                        execute_dict = json.loads(response_str)
                    except json.JSONDecodeError:
                        self._log(
                            f"  - 플레이어 {p.id}: 잘못된 JSON 응답 수신. 찬반 투표를 건너뜁니다."
                        )
                        continue
                    agree_execution = execute_dict.get("agree_execution")
                    if agree_execution == 1:
                        self.final_vote += 1
                        speech = f"- 플레이어 {p.id}이(가) 처형에 찬성합니다."
                    elif agree_execution == -1:
                        self.final_vote -= 1
                        speech = f"- 플레이어 {p.id}이(가) 처형에 반대합니다."
                    else:
                        speech = f"- 플레이어 {p.id}이(가) 처형에 기권합니다."
                    self._log(speech)
                    structured_executes.append(
                        {
                            "target_id": executed_target,
                            "speech": speech,
                        }
                    )
                if self.final_vote > 0:
                    self.players[executed_target].alive = False
                    self._log(f"  - 투표 결과: 찬성 {self.final_vote}표")
                    self._log(f"  - {executed_target}번 플레이어가 처형되었습니다.")

                    # === ROLE REVEAL: Show team alignment (Citizen Team vs Mafia Team) ===
                    executed_role = self.players[executed_target].role
                    if executed_role == config.ROLE_MAFIA:
                        self._log(
                            f"  - [공개] {executed_target}번은 마피아 팀이었습니다!"
                        )
                    else:
                        self._log(
                            f"  - [공개] {executed_target}번은 시민 팀이었습니다."
                        )
                else:
                    self._log(f"  - 투표 결과: 찬성 {self.final_vote}표 (과반 미달)")
                    self._log(
                        f"  - {executed_target}번 플레이어는 처형되지 않았습니다."
                    )
                    self.last_execution_result = None

        self._update_alive_status()

    def _process_night(self):
        mafia_target = None
        doctor_target = None
        police_target = None

        for p in self.players:
            if not p.alive:
                continue
            if p.role == config.ROLE_CITIZEN:
                continue
            night_dict = self._get_game_status(p.id)
            night_dict["living_players"] = [
                i for i, alive in enumerate(self.alive_status) if alive
            ]
            conversation_log_str = ""  # No conversation log for night actions
            response_str = p.get_action(night_dict, conversation_log_str)
            try:
                night_dict = json.loads(response_str)
                target = night_dict.get("target_id")
            except json.JSONDecodeError:
                self._log(
                    f"  - 플레이어 {p.id}: 잘못된 JSON 응답 수신. 밤 행동을 건너뜁니다."
                )
                continue
            # 유효한 타겟인지 검증 (살아있는 플레이어인가?) [cite: 159]
            if (
                target is not None
                and 0 <= target < len(self.players)
                and self.players[target].alive
            ):
                if p.role == config.ROLE_MAFIA:
                    mafia_target = target
                elif p.role == config.ROLE_DOCTOR:
                    doctor_target = target
                elif p.role == config.ROLE_POLICE:
                    police_target = target

        self._log("- 밤 행동 결과:")
        no_death = True
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
                no_death = False
            else:
                self._log(f"  - 의사가 {doctor_target}번 플레이어를 살려냈습니다.")

        # 경찰 조사 결과 저장
        if police_target is not None:
            role_name = self.role_names.get(self.players[police_target].role, "알 수 없음")
            self._log(f"  - 경찰 조사: {police_target}번은 [{role_name}]입니다.")
            self.police_logs[police_target] = self.players[police_target].role

        self.last_night_result = {
            "no_death": no_death,
            "victim": mafia_target if not no_death else None,
        }
        self._update_alive_status()

    # (이하 _update_alive_status, _get_game_status, check_game_over 메서드는 Agent 참조만 수정해서 사용)
    def _update_alive_status(self):
        self.alive_status = [1 if p.alive else 0 for p in self.players]

    def _get_game_status(self, viewer_id: Optional[int] = None) -> Dict:
        is_admin = viewer_id is None
        viewer = self.players[viewer_id] if not is_admin else None
        status = {
            "day": self.day_count,
            "phase": self.phase,
            "alive_status": self.alive_status,
            "last_execution_result": self.last_execution_result,
            "last_night_result": self.last_night_result,
            "vote_records": self.vote_counts,
        }
        if is_admin:
            status.update(
                {
                    "my_role": "ADMIN",
                    "all_player_roles": {
                        p.id: p.role for p in self.players
                    },  # 전체 역할 명단
                    "mafia_team_members": [
                        p.id for p in self.players if p.role == config.ROLE_MAFIA
                    ],
                    "police_investigation_results": self.police_logs,
                }
            )
        else:
            status.update(
                {
                    "my_role": viewer.role,
                    "mafia_team_members": (
                        [p.id for p in self.players if p.role == config.ROLE_MAFIA]
                        if viewer.role == config.ROLE_MAFIA
                        else None
                    ),
                    "police_investigation_results": (
                        self.police_logs if viewer.role == config.ROLE_POLICE else None
                    ),
                }
            )

        return status

    def check_game_over(self) -> Tuple[bool, bool]:
        mafia_count = sum(
            1 for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        )
        citizen_count = sum(
            1 for p in self.players if p.role != config.ROLE_MAFIA and p.alive
        )

        # 무승부 조건: 최대 턴 수 초과
        if self.day_count > config.MAX_DAYS:
            self._log(f"\n게임 종료: {config.MAX_DAYS}일이 지나 무승부입니다!")
            return True, False  # 무승부는 패배로 처리

        if mafia_count == 0:
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            return True, True
        elif mafia_count >= citizen_count:
            self._log(f"\n게임 종료: 마피아 승리!")
            # [변경]
            return True, False

        return False, False
