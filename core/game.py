from typing import List, Dict, Tuple, Optional
from config import config, Role, Phase, EventType
from state import GameStatus, GameEvent, PlayerStatus
import random
import json

# Import RationalCharacter directly
from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent


class MafiaGame:
    def __init__(self, log_file=None):
        # players 리스트에는 RationalCharacter가 들어갑니다.
        self.players: List[LLMAgent] = []
        self.phase = Phase.DAY_DISCUSSION  # Start with discussion phase
        self.day_count = 1
        self.vote_counts = []
        self.log_file = log_file
        self.final_vote = 0
        self.action_history: List[GameEvent] = [] # 게임 전체 이벤트 기록

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> GameStatus:
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = Phase.DAY_DISCUSSION  # Start with discussion phase
        self.players = []
        self.action_history = []

        # 1. 플레이어 생성 (All use LLMAgent)
        for i in range(config.game.PLAYER_COUNT):
            # Create rational agent for all players
            player = LLMAgent(player_id=i)
            self.players.append(player)

        # 2. 역할 할당
        roles_to_assign = config.game.DEFAULT_ROLES.copy()
        random.shuffle(roles_to_assign)

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int

            # 로그 출력
            role_name = Role(role_int).name
            self._log(f"플레이어 {i}: {role_name} (LLM Agent)")

        return self.get_game_status()

    def process_turn(
        self, action: int, ai_claim_role: int = -1
    ) -> Tuple[GameStatus, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self.get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase.name}]")

        # Merged phases: PHASE_DAY_CLAIM removed
        if self.phase == Phase.DAY_DISCUSSION:
            self._process_day_discussion()
            self.phase = Phase.DAY_VOTE
        elif self.phase == Phase.DAY_VOTE:
            self._process_day_vote()
            self.phase = Phase.DAY_EXECUTE
        elif self.phase == Phase.DAY_EXECUTE:
            self._process_day_execute()
            self.phase = Phase.NIGHT
        elif self.phase == Phase.NIGHT:
            self._process_night()
            self.phase = Phase.DAY_DISCUSSION
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self.get_game_status(), is_over, is_win

    def _process_day_discussion(self):
        self._log("  - 낮 토론 시작")
        structured_claims = []

        MAX_DEBATE_ROUNDS = 2
        for debate_round in range(MAX_DEBATE_ROUNDS):
            for p in self.players:
                if not p.alive:
                    continue

                status_obj = self.get_game_status(p.id)
                p.observe(status_obj)
                
                # 대화 기록을 넘길 때 화자 ID를 포함한 JSON 전달
                conversation_log_str = json.dumps(structured_claims, ensure_ascii=False)

                response_str = p.get_action(conversation_log_str)
                try:
                    claim_dict = json.loads(response_str)
                except:
                    continue

                if claim_dict.get("discussion_status") == "End":
                    break

                claim = claim_dict.get("claim")
                role_id = claim_dict.get("role")
                target_id = claim_dict.get("target_id")
                role_name = Role(role_id).name if role_id is not None else "알 수 없음"
                discussion_ended = claim_dict.get("discussion_status")
                reason = claim_dict.get("reason", "")

                # 발언 로깅
                if claim == 0:  # 본인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {role_name}이라 주장합니다."
                    event_type = EventType.CLAIM
                elif claim == 1:  # 타인의 직업 주장
                    speech = f"- 플레이어 {p.id}이(가) {target_id}는 {role_name}라고 주장합니다!"
                    event_type = EventType.CLAIM
                else:
                    speech = f"- 플레이어 {p.id}이(가) 침묵합니다."
                    event_type = EventType.CLAIM # 침묵도 일종의 발언으로 처리하거나 별도 처리

                self._log(speech)
                self._log(f"  - 플레이어 {p.id}의 주장 이유: {reason}")
                # structured_claims에 추가
                structured_claims.append(
                    {
                        "speech": speech,
                        "claim": claim,
                    }
                )
                
                # Action History에 추가
                self.action_history.append(GameEvent(
                    day=self.day_count,
                    phase=self.phase,
                    event_type=event_type,
                    actor_id=p.id,
                    target_id=target_id,
                    value=Role(role_id) if role_id is not None else None # 주장한 역할 (Role Enum)
                ))

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
            
            status_obj = self.get_game_status(p.id)
            p.observe(status_obj)
            
            conversation_log_str = json.dumps(structured_votes, ensure_ascii=False)

            # get_action 호출 및 JSON 파싱
            response_str = p.get_action(conversation_log_str)
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

                # self.players[target].voted_by_last_turn.append(p.id) # Removed legacy
                # self.players[target].vote_history[p.id] += 1 # Removed legacy
                
                self.action_history.append(GameEvent(
                    day=self.day_count,
                    phase=self.phase,
                    event_type=EventType.VOTE,
                    actor_id=p.id,
                    target_id=target,
                    value=None # 투표 사유 제거
                ))
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
            else:
                executed_target = executed_targets[0]
                for p in self.players:
                    if not p.alive:
                        continue
                    
                    status_obj = self.get_game_status(p.id)
                    
                    conversation_log_str = json.dumps(
                        structured_executes + [{"execution_target": executed_target}], ensure_ascii=False
                    )
                    
                    p.observe(status_obj)
                    
                    # get_action 호출 및 JSON 파싱
                    response_str = p.get_action(conversation_log_str)
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
                    
                    # 1. 처형 성공 이벤트 (value=True)
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.EXECUTE,
                        actor_id=-1, # System
                        target_id=executed_target,
                        value=True
                    ))
                    
                    # 2. 직업 공개 이벤트 (POLICE_RESULT by System)
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.POLICE_RESULT,
                        actor_id=-1, # System
                        target_id=executed_target,
                        value=executed_role
                    ))

                    if executed_role == Role.MAFIA:
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
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.EXECUTE,
                        actor_id=-1,
                        target_id=executed_target,
                        value=False # 실패
                    ))

        self._update_alive_status()

    def _process_night(self):
        mafia_target = None
        doctor_target = None
        police_target = None

        for p in self.players:
            if not p.alive:
                continue
            if p.role == Role.CITIZEN:
                continue
            
            status_obj = self.get_game_status(p.id)
            p.observe(status_obj)
            
            conversation_log_str = ""  # No conversation log for night actions
            response_str = p.get_action(conversation_log_str)
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
                if p.role == Role.MAFIA:
                    mafia_target = target
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.KILL,
                        actor_id=p.id,
                        target_id=target,
                        value=None
                    ))
                elif p.role == Role.DOCTOR:
                    doctor_target = target
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.PROTECT,
                        actor_id=p.id,
                        target_id=target,
                        value=None
                    ))
                elif p.role == Role.POLICE:
                    police_target = target
                    # 경찰 결과는 즉시 알 수 있음 (다음 턴 observe에서 반영됨)
                    self.action_history.append(GameEvent(
                        day=self.day_count,
                        phase=self.phase,
                        event_type=EventType.POLICE_RESULT,
                        actor_id=p.id,
                        target_id=target,
                        value=self.players[target].role
                    ))

        self._log("- 밤 행동 결과:")
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
            else:
                self._log(f"  - 의사가 {doctor_target}번 플레이어를 살려냈습니다.")

        # 경찰 조사 결과 저장 (로그만 출력, 데이터는 action_history에 있음)
        if police_target is not None:
            role_name = Role(self.players[police_target].role).name
            self._log(f"  - 경찰 조사: {police_target}번은 [{role_name}]입니다.")

    def get_game_status(self, viewer_id: Optional[int] = None) -> GameStatus:
        # Admin view if viewer_id is None
        if viewer_id is None:
            # Admin sees everything
            return GameStatus(
                day=self.day_count,
                phase=self.phase,
                my_id=-1,
                my_role=Role.CITIZEN, # Dummy
                players=[PlayerStatus(id=p.id, alive=p.alive) for p in self.players],
                action_history=self.action_history
            )

        viewer = self.players[viewer_id]
        
        player_statuses = []
        for p in self.players:
            player_statuses.append(PlayerStatus(
                id=p.id,
                alive=p.alive
            ))

        filtered_history = []
        
        # [Mafia Team Info] Inject teammate info as system events
        if viewer.role == Role.MAFIA:
            for p in self.players:
                if p.role == Role.MAFIA and p.id != viewer_id:
                    filtered_history.append(GameEvent(
                        day=0, # Game Start
                        phase=Phase.DAY_DISCUSSION,
                        event_type=EventType.POLICE_RESULT, # Treat as investigation result
                        actor_id=-1, # System
                        target_id=p.id,
                        value=Role.MAFIA
                    ))

        for event in self.action_history:
            if event.phase == Phase.NIGHT:
                if event.actor_id == viewer_id:
                    filtered_history.append(event)
                elif viewer.role == Role.MAFIA and self.players[event.actor_id].role == Role.MAFIA:
                    filtered_history.append(event)
                elif event.event_type == EventType.POLICE_RESULT:
                    if event.actor_id == viewer_id:
                        filtered_history.append(event)
            else:
                filtered_history.append(event)

        return GameStatus(
            day=self.day_count,
            phase=self.phase,
            my_id=viewer_id,
            my_role=viewer.role,
            players=player_statuses,
            action_history=filtered_history
        )

    def check_game_over(self) -> Tuple[bool, bool]:
        mafia_count = sum(
            1 for p in self.players if p.role == Role.MAFIA and p.alive
        )
        citizen_count = sum(
            1 for p in self.players if p.role != Role.MAFIA and p.alive
        )

        # 무승부 조건: 최대 턴 수 초과
        if self.day_count > config.game.MAX_DAYS:
            self._log(f"\n게임 종료: {config.game.MAX_DAYS}일이 지나 무승부입니다!")
            return True, False  # 무승부는 패배로 처리

        if mafia_count == 0:
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            return True, True
        elif mafia_count >= citizen_count:
            self._log(f"\n게임 종료: 마피아 승리!")
            return True, False

        return False, False
