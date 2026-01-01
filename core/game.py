from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType, ActionType
from state import GameStatus, GameEvent, PlayerStatus, GameAction
from core.agent.llmAgent import LLMAgent
from core.agent.baseAgent import BaseAgent
from core.logger import LogManager


class MafiaGame:
    def __init__(
        self, log_file=None, agents: Optional[List[BaseAgent]] = None, logger=None
    ):
        """
        Args:
            log_file: (Deprecated) 레거시 로그 파일 핸들 - 하위 호환성을 위해 유지
            agents: 외부에서 주입받을 에이전트 리스트 (선택적)
            logger: LogManager 인스턴스 (선택적)
        """
        self.players: List[BaseAgent] = agents if agents is not None else []
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.history: List[GameEvent] = []
        self.log_file = log_file
        self.logger = logger

    def reset(self, agents: Optional[List[BaseAgent]] = None) -> GameStatus:
        """
        Args:
            agents: 외부에서 주입받을 에이전트 리스트 (선택적)
        """
        self.day = 1
        self.phase = Phase.GAME_START
        self.history = []
        self._last_votes = []

        if agents is not None:
            self.players = agents
        elif not self.players:
            self.players = [
                LLMAgent(player_id=i, logger=self.logger)
                for i in range(config.game.PLAYER_COUNT)
            ]

        roles = config.game.DEFAULT_ROLES.copy()
        random.shuffle(roles)
        for p, r in zip(self.players, roles):
            p.role = r
            # 역할 할당 이벤트 로깅
            if self.logger:
                # GameEvent로 기록 (JSONL 저장용)

                event = GameEvent(
                    day=0,
                    phase=Phase.GAME_START,
                    event_type=EventType.SYSTEM_MESSAGE,  # 역할 공개 이벤트로 활용
                    actor_id=-1,
                    target_id=p.id,
                    value=p.role,
                )
                self.logger.log_event(event)

        return self.get_game_status()

    def process_turn(self) -> Tuple[GameStatus, bool, bool]:
        """
        게임 턴 진행 - 외부에서 액션을 주입받지 않고 에이전트에게 직접 의사를 물음
        """
        is_over, is_win = self.check_game_over()
        if is_over:
            return self.get_game_status(), is_over, is_win

        if self.phase == Phase.GAME_START:
            self.phase = Phase.DAY_DISCUSSION
        elif self.phase == Phase.DAY_DISCUSSION:
            self._process_discussion()
            self.phase = Phase.DAY_VOTE
        elif self.phase == Phase.DAY_VOTE:
            self._process_vote()
            self.phase = Phase.DAY_EXECUTE
        elif self.phase == Phase.DAY_EXECUTE:
            self._process_execute()
            self.phase = Phase.NIGHT
        elif self.phase == Phase.NIGHT:
            self._process_night()
            self.phase = Phase.DAY_DISCUSSION
            self.day += 1

        is_over, is_win = self.check_game_over()
        return self.get_game_status(), is_over, is_win

    def _process_discussion(self):
        """
        토론 단계 처리 - 동시성 구조
        모든 플레이어가 동시에 발언을 결정하고, 한 번에 공개
        """
        for round_num in range(2):  # 최대 2라운드
            alive_count = sum(1 for p in self.players if p.alive)

            # Phase 1: 모든 플레이어가 동시에 행동 결정
            actions = []
            for p in self.players:
                if not p.alive:
                    continue

                try:
                    # 현재 상태만 보고 행동 결정 (다른 플레이어의 이번 행동은 모름)
                    action = p.get_action()

                    if not isinstance(action, GameAction):
                        print(
                            f"[Engine] Warning: Player {p.id} returned non-MafiaAction: {type(action)}"
                        )
                        continue

                    actions.append((p.id, action))

                except Exception as e:
                    print(f"[Engine] Error processing action for Player {p.id}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            # Phase 2: 모든 행동을 한 번에 처리 (이벤트 생성)
            pass_count = 0
            for player_id, action in actions:
                print(f"[Engine] Player {player_id} action: {action}")

                # PASS 카운트
                if action.action_type == ActionType.PASS:
                    pass_count += 1

                # CLAIM 이벤트만 기록
                if action.action_type == ActionType.CLAIM:
                    event = GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.CLAIM,
                        actor_id=player_id,
                        target_id=action.target_id if action.target_id != -1 else None,
                        value=action.claim_role,
                    )

                    self.history.append(event)
                    if self.logger:
                        self.logger.log_event(event)

            # Phase 3: 결과를 모두에게 공개
            for p in self.players:
                if p.alive:
                    p.observe(self.get_game_status(p.id))

            # 전원 침묵 시에만 토론 종료
            if pass_count >= alive_count:
                print(
                    f"[Engine] Discussion ended: all {alive_count} players passed in round {round_num + 1}"
                )
                break

    def _process_vote(self):
        """
        투표 단계 처리 - 동시성 구조
        모든 플레이어가 동시에 투표하고, 한 번에 집계
        """
        votes = [0] * len(self.players)

        # Phase 1: 모든 플레이어가 동시에 투표 결정
        vote_actions = []
        for p in self.players:
            if not p.alive:
                continue

            # 에이전트로부터 MafiaAction 받기
            action = p.get_action()

            if not isinstance(action, GameAction):
                print(
                    f"[Engine] Warning: Player {p.id} returned non-MafiaAction in vote"
                )
                continue

            vote_actions.append((p.id, action))

        # Phase 2: 모든 투표를 한 번에 처리
        for player_id, action in vote_actions:
            target_id = action.target_id

            # 유효한 투표인지 확인
            if (
                target_id != -1
                and 0 <= target_id < len(self.players)
                and self.players[target_id].alive
            ):
                votes[target_id] += 1

                # 투표 이벤트 생성
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.VOTE,
                    actor_id=player_id,
                    target_id=target_id,
                )
                self.history.append(event)
                if self.logger:
                    self.logger.log_event(event)

        self._last_votes = votes

        # Phase 3: 결과를 모두에게 공개
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_execute(self):
        """
        처형 단계 처리 - 동시성 구조
        최다 득표자에 대한 처형 동의를 동시에 받고 처리
        """
        if not self._last_votes:
            return

        max_v = max(self._last_votes)
        if max_v > 0:
            targets = [i for i, v in enumerate(self._last_votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]

                # Phase 1: 모든 플레이어가 동시에 처형 동의 여부 결정
                execution_votes = []
                for p in self.players:
                    if not p.alive:
                        continue

                    # 에이전트로부터 MafiaAction 받기
                    action = p.get_action()

                    execution_votes.append((p.id, action))

                # Phase 2: 모든 동의를 한 번에 처리
                final_score = 0
                for player_id, action in execution_votes:
                    # 처형 동의 여부 확인 (임시: dict 호환성 유지)
                    if isinstance(action, dict):
                        agree = action.get("agree_execution", 0)
                        final_score += agree
                    elif isinstance(action, GameAction):
                        # MafiaAction의 경우 target_id가 처형 대상과 일치하면 동의로 간주
                        if action.target_id == target_id:
                            final_score += 1

                success = final_score > 0
                if success:
                    self.players[target_id].alive = False

                # 처형 이벤트 생성 및 로깅
                execute_event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.EXECUTE,
                    actor_id=-1,
                    target_id=target_id,
                    value=self.players[target_id].role if success else None,
                )
                self.history.append(execute_event)
                if self.logger:
                    self.logger.log_event(execute_event)

                # 경찰 결과 이벤트 (처형 성공 시 역할 공개)
                if success:
                    police_result_event = GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.SYSTEM_MESSAGE,
                        actor_id=-1,
                        target_id=target_id,
                        value=self.players[target_id].role,
                    )
                    self.history.append(police_result_event)
                    if self.logger:
                        self.logger.log_event(police_result_event)

        # Phase 3: 결과를 모두에게 공개
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_night(self):
        """
        밤 단계 처리 - 동시성 구조
        모든 역할이 동시에 행동을 결정하고, 한 번에 처리
        """
        # 1. 역할별 행동 설정 매핑
        role_config = {
            Role.MAFIA: {"event_type": EventType.KILL, "value": None},
            Role.DOCTOR: {"event_type": EventType.PROTECT, "value": None},
            Role.POLICE: {
                "event_type": EventType.POLICE_RESULT,
                "value": "role_of_target",
            },
        }

        # Phase 1: 모든 플레이어가 동시에 밤 행동 결정
        night_actions = []
        for p in self.players:
            if not p.alive or p.role == Role.CITIZEN:
                continue

            # 에이전트로부터 MafiaAction 받기
            action = p.get_action()

            if not isinstance(action, GameAction):
                print(
                    f"[Engine] Warning: Player {p.id} returned non-MafiaAction in night"
                )
                continue

            night_actions.append((p.id, p.role, action))

        # Phase 2: 모든 밤 행동을 한 번에 처리
        night_targets = {}
        for player_id, role, action in night_actions:
            target_id = action.target_id

            # 유효한 타겟인지 확인
            if (
                target_id != -1
                and 0 <= target_id < len(self.players)
                and self.players[target_id].alive
            ):
                # 역할별 설정 가져오기
                config = role_config.get(role)
                if not config:
                    continue

                # 타겟 저장 (살해/보호 판정용)
                night_targets[role] = target_id

                # 이벤트 생성
                event_value = None
                if config["value"] == "role_of_target":
                    event_value = self.players[target_id].role

                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=config["event_type"],
                    actor_id=player_id,
                    target_id=target_id,
                    value=event_value,
                )
                self.history.append(event)
                if self.logger:
                    self.logger.log_event(event)

        # Phase 3: 살해 및 보호 처리 (동시 발생)
        mafia_target = night_targets.get(Role.MAFIA)
        doctor_target = night_targets.get(Role.DOCTOR)

        if mafia_target is not None and mafia_target != doctor_target:
            self.players[mafia_target].alive = False

        # Phase 4: 결과를 모두에게 공개
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def get_game_status(self, viewer_id: Optional[int] = None) -> GameStatus:
        if viewer_id is None:
            return GameStatus(
                day=self.day,
                phase=self.phase,
                my_id=-1,
                my_role=Role.CITIZEN,
                players=[PlayerStatus(id=p.id, alive=p.alive) for p in self.players],
                action_history=self.history,
            )

        viewer = self.players[viewer_id]
        filtered = []
        if viewer.role == Role.MAFIA:
            for p in self.players:
                if p.role == Role.MAFIA and p.id != viewer_id:
                    filtered.append(
                        GameEvent(
                            day=0,
                            phase=Phase.DAY_DISCUSSION,
                            event_type=EventType.POLICE_RESULT,
                            actor_id=-1,
                            target_id=p.id,
                            value=Role.MAFIA,
                        )
                    )

        for e in self.history:
            if e.phase == Phase.NIGHT:
                if (
                    e.actor_id == viewer_id
                    or (
                        viewer.role == Role.MAFIA
                        and self.players[e.actor_id].role == Role.MAFIA
                    )
                    or (
                        e.event_type == EventType.POLICE_RESULT
                        and e.actor_id == viewer_id
                    )
                ):
                    filtered.append(e)
            else:
                filtered.append(e)

        return GameStatus(
            day=self.day,
            phase=self.phase,
            my_id=viewer_id,
            my_role=viewer.role,
            players=[PlayerStatus(id=p.id, alive=p.alive) for p in self.players],
            action_history=filtered,
        )

    def check_game_over(self) -> Tuple[bool, bool]:
        m_count = sum(1 for p in self.players if p.role == Role.MAFIA and p.alive)
        c_count = sum(1 for p in self.players if p.role != Role.MAFIA and p.alive)
        if self.day > config.game.MAX_DAYS:
            return True, False
        if m_count == 0:
            return True, True
        if m_count >= c_count:
            return True, False
        return False, False
