from typing import List, Dict, Tuple
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

        role_names = {
            v: k.replace("ROLE_", "")
            for k, v in config.__dict__.items()
            if k.startswith("ROLE_")
        }

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int

            # 로그 출력
            role_name = role_names.get(role_int, "Unknown")
            self._log(f"플레이어 {i}: {role_name} (LLM Agent)")

        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]
        return self._get_game_status()

    def process_turn(
        self, action: int, ai_claim_role: int = -1
    ) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win

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
        self._log("  - 낮 토론: 플레이어들이 의견을 나누고 의심도를 갱신합니다.")
        structured_claims = []  # Store all claims in new dictionary format

        MAX_DEBATE_ROUNDS = 5  # 최대 토론 라운드 수
        for debate_round in range(MAX_DEBATE_ROUNDS):
            self._log(f"  - 토론 라운드 {debate_round + 1}")
            discussion_ended = False
            # Agent들의 주장
            for p in self.players:
                if not p.alive:
                    continue

                # LLMAgent.get_action에 맞는 파라미터 전달
                game_state_dict = self._get_game_status()
                conversation_log_str = json.dumps(structured_claims, ensure_ascii=False)

                # get_action 호출 및 JSON 파싱
                response_str = p.get_action(game_state_dict, conversation_log_str)
                try:
                    claim_dict = json.loads(response_str)
                except json.JSONDecodeError:
                    self._log(
                        f"  - 플레이어 {p.id}: 잘못된 JSON 응답 수신. 토론을 건너뜁니다."
                    )
                    continue

                # 토론 종료 제안 확인
                if claim_dict.get("discussion_status") == "End":
                    self._log(f"  - 플레이어 {p.id}이(가) 토론 종료를 제안합니다.")
                    discussion_ended = True
                    break

                claim = claim_dict.get("claim")
                role_id = claim_dict.get("role")
                target_id = claim_dict.get("target_id")
                role_name = {0: "시민", 1: "경찰", 2: "의사", 3: "마피아"}.get(
                    role_id, "알 수 없음"
                )

                # 발언 로깅
                if claim == 0:  # 본인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {role_name}이라 주장합니다."
                elif claim == 1:  # 타인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {target_id}는 {role_name}라고 주장합니다!"
                else:
                    speech = f"- 플레이어 {p.id}이(가) 침묵합니다."

                self._log(speech)

                # structured_claims에 추가
                structured_claims.append(
                    {
                        "speech": speech,
                        "claim": claim,
                    }
                )

            if discussion_ended:
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
            game_state_dict = self._get_game_status()
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
            speech = vote_dict.get("reasoning")

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
                    game_state_dict = self._get_game_status()
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
            night_dict = self._get_game_status()
            conversation_log_str = ""  # No conversation log for night actions
            response_str = p.get_action(night_dict, conversation_log_str)
            try:
                night_dict = json.loads(response_str)
            except json.JSONDecodeError:
                self._log(
                    f"  - 플레이어 {p.id}: 잘못된 JSON 응답 수신. 밤 행동을 건너뜁니다."
                )
                continue
            target = night_dict.get("target_id")
            if p.role == config.ROLE_MAFIA:
                mafia_target = target
            elif p.role == config.ROLE_DOCTOR:
                doctor_target = target
            elif p.role == config.ROLE_POLICE:
                police_target = target
        self._log("- 밤 행동 결과:")
        # 결과 정산
        no_death = False
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
                no_death = False
            else:
                self._log(f"  - 의사가 {doctor_target}을(를) 살려냈습니다.")
                no_death = True
        
        # 경찰 조사 결과 로그 (경찰이 있고, 대상을 지정했을 때만)
        if police_target is not None:
            # 역할 이름 찾기
            role_names = {
                v: k.replace("ROLE_", "")
                for k, v in config.__dict__.items()
                if k.startswith("ROLE_")
            }
            role_name = role_names.get(self.players[police_target].role, "알 수 없음")
            self._log(
                f"  - 경찰의 조사 결과: {police_target}번 플레이어는 [{role_name}]입니다."
            )

        # Store night result for belief updates (Doctor's Logic)
        self.last_night_result = {
            "no_death": no_death,
            "last_healed": doctor_target if doctor_target is not None else -1,
        }

        self._update_alive_status()

    # (이하 _update_alive_status, _get_game_status, check_game_over 메서드는 Agent 참조만 수정해서 사용)
    def _update_alive_status(self):
        self.alive_status = [1 if p.alive else 0 for p in self.players]

    def _get_game_status(self) -> Dict:
        return {
            "day": self.day_count,
            "phase": self.phase,
            "alive_status": self.alive_status,
            "roles": [p.role for p in self.players],
        }

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
            return False, False

        return False, False
