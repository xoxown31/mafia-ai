from typing import List, Dict, Tuple
import config
import random

# Import LLMAgent directly
from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent



class MafiaGame:
    def __init__(self, log_file=None):
        self.players: List[LLMAgent] = []
        self.phase = config.PHASE_DAY_DISCUSSION
        self.day_count = 1
        self.alive_status = []
        self.vote_counts = []
        self.log_file = log_file
        self.final_vote = 0
        self.last_execution_result = None
        self.last_night_result = None
        self.ai_id = -1

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> Dict:
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = config.PHASE_DAY_DISCUSSION
        self.players = []
        self.last_execution_result = None
        self.last_night_result = None

        # AI 플레이어 ID 무작위 선정 (환경과의 연동을 위해 reset에서 수행)
        self.ai_id = random.randint(0, config.PLAYER_COUNT - 1)
        self._log(f"[System] 이번 게임의 AI 플레이어는 {self.ai_id}번 입니다.")

        for i in range(config.PLAYER_COUNT):
            # 모든 플레이어를 LLMAgent(또는 개편될 BaseAgent 자식들)로 생성
            player = LLMAgent(player_id=i)
            self.players.append(player)

        roles_to_assign = config.ROLES.copy()
        random.shuffle(roles_to_assign)

        role_names = {
            config.ROLE_CITIZEN: "CITIZEN",
            config.ROLE_POLICE: "POLICE",
            config.ROLE_DOCTOR: "DOCTOR",
            config.ROLE_MAFIA: "MAFIA",
        }

        mafia_ids = [i for i, r in enumerate(roles_to_assign) if r == config.ROLE_MAFIA]

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int

            # [수정] 마피아는 팀원을 마피아(Role 3)로 확신하고 시민(Role 0)이 아님을 인지
            if role_int == config.ROLE_MAFIA:
                for partner_id in mafia_ids:
                    if partner_id != i:
                        self.players[i].belief[partner_id, config.ROLE_MAFIA] = 100.0
                        self.players[i].belief[partner_id, config.ROLE_CITIZEN] = -100.0

            # 로그 출력
            role_name = role_names.get(role_int, "Unknown")
            self._log(f"플레이어 {i}: {role_name} (Agent)")

        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]
        return self._get_game_status()

    def process_turn(
        self, action: int, ai_claim_role: int = -1
    ) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        # AI 에이전트 객체에 현재 턴의 RL 액션 주입
        if 0 <= self.ai_id < len(self.players) and self.players[self.ai_id].alive:
            # 개편될 Agent 클래스들은 set_turn_action을 통해 외부 액션을 수용함
            if hasattr(self.players[self.ai_id], "set_turn_action"):
                self.players[self.ai_id].set_turn_action(action, ai_claim_role)

        # 게임 페이즈 진행
        if self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion()
            self.phase = config.PHASE_DAY_VOTE
        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote()
            self.phase = config.PHASE_NIGHT
        elif self.phase == config.PHASE_NIGHT:
            self._process_night()
            self.phase = config.PHASE_DAY_DISCUSSION
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def _process_day_discussion(self):
        """낮 토론 및 역할 주장 처리 (다중 라운드)"""
        self._log("  - 낮 토론: 플레이어들이 의견을 나누고 의심도를 갱신합니다.")
        all_claims = []
        MAX_DEBATE_ROUNDS = 5

        for round_idx in range(MAX_DEBATE_ROUNDS):
            round_claims = []
            for p in self.players:
                if not p.alive:
                    continue

                # get_action을 통해 토론 발언(Claim) 수집
                claim = p.get_action(self.phase, context=all_claims)
                if claim and claim.get("type") == "CLAIM":
                    claim["speaker_id"] = p.id
                    round_claims.append(claim)
                    all_claims.append(claim)

                    # 발언 로그 기록
                    target = claim.get("target_id")
                    assertion = claim.get("assertion", "SUSPECT")
                    self._log(
                        f"  - 플레이어 {p.id}가 {target}번을 {assertion}으로 지목했습니다."
                    )

            if not round_claims:  # 더 이상 새로운 발언이 없으면 종료
                break

            # 발언 내용을 바탕으로 모든 에이전트의 Belief 업데이트
            game_status = self._create_game_status_for_update(claims=round_claims)
            for p in self.players:
                if p.alive:
                    p.update_belief(game_status)

    def _process_day_vote(self):
        """투표 및 처형 집행"""
        self.vote_counts = [0] * config.PLAYER_COUNT
        voters_map = {}

        for p in self.players:
            if not p.alive:
                continue

            # get_action을 통해 투표 대상 ID 수집
            target = p.get_action(self.phase)
            if 0 <= target < config.PLAYER_COUNT:
                self.vote_counts[target] += 1
                voters_map.setdefault(target, []).append(p.id)
                self._log(f"  - 플레이어 {p.id}가 {target}번에게 투표했습니다.")
            else:
                self._log(f"  - 플레이어 {p.id}가 기권했습니다.")

        # 최다 득표자 판별 및 처형
        max_votes = max(self.vote_counts) if self.vote_counts else 0
        if max_votes > 0:
            candidates = [i for i, v in enumerate(self.vote_counts) if v == max_votes]
            if len(candidates) == 1:
                executed_id = candidates[0]
                role = self.players[executed_id].role
                team = "MAFIA" if role == config.ROLE_MAFIA else "CITIZEN"

                self.players[executed_id].alive = False
                self._log(f"  - 투표 결과: {executed_id}번 처형 (팀: {team})")

                # 처형 결과 저장 (update_belief용)
                vote_info = {
                    "voters": voters_map.get(executed_id, []),
                    "target": executed_id,
                }
                self.last_execution_result = (
                    executed_id,
                    team,
                    self.day_count,
                    vote_info,
                )
            else:
                self._log(
                    f"  - 투표 결과: 동률 발생({candidates}), 처형이 무산되었습니다."
                )
                self.last_execution_result = None

        # 결과 반영을 위한 전원 Belief 업데이트
        game_status = self._create_game_status_for_update()
        for p in self.players:
            if p.alive:
                p.update_belief(game_status)

    def _process_night(self):
        """밤 행동(살해, 치료, 조사) 처리"""
        mafia_target = -1
        doctor_target = -1
        police_results = {}

        # 1. 행동 수집
        for p in self.players:
            if not p.alive:
                continue

            action = p.get_action(self.phase)
            if p.role == config.ROLE_MAFIA:
                mafia_target = action  # (참고: 마피아가 여러 명일 경우 합의 로직 필요)
            elif p.role == config.ROLE_DOCTOR:
                doctor_target = action
            elif p.role == config.ROLE_POLICE:
                if action != -1:
                    is_mafia = self.players[action].role == config.ROLE_MAFIA
                    # 결과를 저장해뒀다가 나중에 알려주거나, 여기서 즉시 주입
                    police_results[p.id] = (action, is_mafia)

        # 2. 살해 및 치료 정산
        no_death = False
        if mafia_target != -1:
            if mafia_target == doctor_target:
                self._log(f"  - [밤] 의사가 {doctor_target}번 플레이어를 살려냈습니다.")
                no_death = True
            else:
                self.players[mafia_target].alive = False
                self._log(
                    f"  - [밤] {mafia_target}번 플레이어가 마피아에게 살해당했습니다."
                )

        for pid, (target_id, is_mafia) in police_results.items():
            # LLMAgent에 _update_police_result 같은 메서드를 만들거나
            # belief를 직접 건드려야 함 (Game은 신이므로 허용)
            p = self.players[pid]
            if p.alive:
                score = 100.0 if is_mafia else -100.0
                p.belief[target_id, config.ROLE_MAFIA] = score
                # 로그 출력
                role_str = "마피아" if is_mafia else "시민"
                self._log(
                    f"  - [경찰] {pid}번이 {target_id}번을 조사하여 '{role_str}'임을 알았습니다."
                )

        # 3. 결과 저장 및 전원 업데이트
        self.last_night_result = {"no_death": no_death, "last_healed": doctor_target}
        game_status = self._create_game_status_for_update()
        for p in self.players:
            if p.alive:
                p.update_belief(game_status)

    def _create_game_status_for_update(self, claims: List = None) -> Dict:
        """에이전트의 update_belief에 전달할 공통 게임 상태 생성"""
        return {
            "claims": claims or [],
            "alive_players": [p.id for p in self.players if p.alive],
            "execution_result": self.last_execution_result,
            "night_result": self.last_night_result,
            "day": self.day_count,
        }

    def _get_game_status(self) -> Dict:
        return {
            "day": self.day_count,
            "phase": self.phase,
            "alive_status": [1 if p.alive else 0 for p in self.players],
            "roles": [p.role for p in self.players],
            "id": self.ai_id,
        }

    def check_game_over(self) -> Tuple[bool, bool]:
        mafia_count = sum(
            1 for p in self.players if p.alive and p.role == config.ROLE_MAFIA
        )
        citizen_count = sum(
            1 for p in self.players if p.alive and p.role != config.ROLE_MAFIA
        )

        if self.day_count > config.MAX_DAYS:
            self._log("게임 종료: 무승부 (최대 일수 초과)")
            return True, False

        if mafia_count == 0:
            self._log("게임 종료: 시민 팀 승리")
            return True, self.players[self.ai_id].role != config.ROLE_MAFIA

        if mafia_count >= citizen_count:
            self._log("게임 종료: 마피아 팀 승리")
            return True, self.players[self.ai_id].role == config.ROLE_MAFIA

        return False, False
