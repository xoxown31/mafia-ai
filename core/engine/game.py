from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType, ActionType
from core.engine.state import GameStatus, GameEvent, PlayerStatus, GameAction
from core.agents.llm_agent import LLMAgent
from core.agents.base_agent import BaseAgent
from core.managers.logger import LogManager
from collections import Counter


class MafiaGame:
    def __init__(self, agents: List[BaseAgent]):
        """
        Args:
            agents: 외부에서 주입받을 에이전트 리스트 (필수)
        """
        self.players: List[BaseAgent] = agents
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.discussion_round = 0
        self.history: List[GameEvent] = []

    def reset(self) -> GameStatus:
        """
        게임 상태 초기화
        """
        self.day = 1
        self.phase = Phase.GAME_START
        self.discussion_round = 0
        self.history = []
        self._last_votes = []  # 투표 결과 초기화
        self.police_research = [0 for _ in range(7)]  # 경찰 조사 결과 초기화

        # 에이전트 상태 초기화 (필요 시)
        # 외부에서 주입된 에이전트를 그대로 사용

        roles = config.game.DEFAULT_ROLES.copy()
        random.shuffle(roles)
        for p, r in zip(self.players, roles):
            p.role = r
            p.vote_count = 0  # 투표 수 초기화
            p.alive = True
            # GameEvent로 기록 (JSONL 저장용)

            event = GameEvent(
                day=0,
                phase=Phase.GAME_START,
                event_type=EventType.SYSTEM_MESSAGE,  # 역할 공개 이벤트로 활용
                actor_id=-1,
                target_id=p.id,
                value=p.role,
            )
            self.history.append(event)

        return self.get_game_status()

    def step_phase(
        self, actions: Dict[int, GameAction]
    ) -> Tuple[GameStatus, bool, bool]:
        """
        게임 페이즈 진행 - 외부에서 주입된 액션 처리
        단일 페이즈(Phase) 단위로 실행을 멈추고 제어권을 반환
        """
        is_over, is_win = self.check_game_over()

        # 게임 종료 시 이벤트 기록 (중복 방지)
        if is_over:
            game_ended_before = any(e.phase == Phase.GAME_END for e in self.history)
            if not game_ended_before:
                event = GameEvent(
                    day=self.day,
                    phase=Phase.GAME_END,
                    event_type=EventType.SYSTEM_MESSAGE,
                    actor_id=-1,
                    target_id=-1,
                    value=is_win,  # 시민 승리 여부
                )
                self.history.append(event)
            return self.get_game_status(), is_over, is_win

        if self.phase == Phase.GAME_START:
            self.phase = Phase.DAY_DISCUSSION
        elif self.phase == Phase.DAY_DISCUSSION:
            phase_end = self._process_discussion(actions)
            if phase_end:
                self.phase = Phase.DAY_VOTE
        elif self.phase == Phase.DAY_VOTE:
            self._process_vote(actions)
            self.phase = Phase.DAY_EXECUTE
        elif self.phase == Phase.DAY_EXECUTE:
            self._process_execute(actions)
            self.phase = Phase.NIGHT
        elif self.phase == Phase.NIGHT:
            self._process_night(actions)
            self.phase = Phase.DAY_DISCUSSION
            self.day += 1

        is_over, is_win = self.check_game_over()
        if is_over:
            game_ended_before = any(e.phase == Phase.GAME_END for e in self.history)
            if not game_ended_before:
                event = GameEvent(
                    day=self.day,
                    phase=Phase.GAME_END,
                    event_type=EventType.SYSTEM_MESSAGE,
                    actor_id=-1,
                    target_id=-1,
                    value=is_win,  # 시민 승리 여부
                )
                self.history.append(event)

        return self.get_game_status(), is_over, is_win

    def _process_discussion(self, actions: Dict[int, GameAction]) -> bool:
        """
        토론 단계 처리 - 외부에서 주입된 액션 처리
        Returns:
            bool: 페이즈 종료 여부 (True면 다음 페이즈로)
        """
        alive_count = sum(1 for p in self.players if p.alive)

        # 순차적 반영: 8명의 액션을 ID 순으로 루프를 돌며 하나씩 적용
        # 이는 각 액션을 GameEvent 시퀀스로 만들어 RNN Hidden State를 갱신하기 위함입니다.
        sorted_actions = sorted(actions.items(), key=lambda x: x[0])

        pass_count = 0
        for player_id, action in sorted_actions:
            player = self.players[player_id]
            if not player.alive:
                continue

            # 1. PASS 처리
            is_pass = action.target_id == -1 and action.claim_role is None
            if is_pass:
                pass_count += 1
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.CLAIM,
                    actor_id=player_id,
                    value=None,
                )

            # 2. CLAIM 처리
            elif action.claim_role is not None:
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.CLAIM,
                    actor_id=player_id,
                    target_id=action.target_id if action.target_id != -1 else None,
                    value=action.claim_role,
                )

            # 3. TARGET_ACTION (토론 중 지목) 처리
            elif action.target_id != -1:
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.CLAIM,
                    actor_id=player_id,
                    target_id=action.target_id,
                    value=None,
                )

            # 생성된 이벤트를 히스토리에 추가 및 로깅
            self.history.append(event)

        # 토론 종료 조건 확인
        if pass_count >= alive_count * 0.5:
            return True

        self.discussion_round += 1
        return self.discussion_round >= config.game.MAX_DISCUSSION_ROUNDS

    def _process_vote(self, actions: Dict[int, GameAction]):
        """
        투표 단계 처리 - 외부에서 주입된 액션 처리
        """

        filtered_actions = {
            player_id: action
            for player_id, action in actions.items()
            if self.players[player_id].alive
        }
        votes = [0] * len(self.players)

        # 투표 수 초기화 (보상 계산 안전성 확보)
        for p in self.players:
            p.vote_count = 0

        # Phase 2: 모든 투표를 한 번에 처리
        for player_id, action in filtered_actions.items():
            target_id = action.target_id

            # 투표 이벤트는 기권(-1)을 포함하여 모두 기록
            event = GameEvent(
                day=self.day,
                phase=self.phase,
                event_type=EventType.VOTE,
                actor_id=player_id,
                target_id=target_id,
            )
            self.history.append(event)

            # 유효한 투표인지 확인 후 집계
            if (
                target_id != -1
                and 0 <= target_id < len(self.players)
                and self.players[target_id].alive
            ):
                votes[target_id] += 1
                self.players[target_id].vote_count += 1

        self._last_votes = votes

        return True

    def _process_execute(self, actions: Dict[int, GameAction]) -> bool:
        """
        처형 단계 처리 - 외부에서 주입된 액션 처리
        """
        if not self._last_votes:
            return True

        max_v = max(self._last_votes)
        execute_event = None

        if max_v > 0:
            targets = [i for i, v in enumerate(self._last_votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]
                filtered_actions = {
                    player_id: action
                    for player_id, action in actions.items()
                    if self.players[player_id].alive
                }

                # Phase 2: 모든 동의를 한 번에 처리
                final_score = 0
                for player_id, action in filtered_actions.items():
                    # 처형 동의 여부 확인 (임시: dict 호환성 유지)
                    if isinstance(action, dict):
                        agree = action.get("agree_execution", 0)
                        final_score += agree
                    elif isinstance(action, GameAction):
                        # GameAction의 경우 target_id가 처형 대상과 일치하면 동의로 간주
                        if action.target_id == target_id:
                            final_score += 1

                success = final_score > 0
                if success:
                    self.players[target_id].alive = False

                # 처형 시도 이벤트 생성
                execute_event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.EXECUTE,
                    actor_id=-1,
                    target_id=target_id,
                    value=self.players[target_id].role if success else None,
                )
            else:
                # 동점으로 처형 무산
                execute_event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.EXECUTE,
                    actor_id=-1,
                    target_id=-1,
                )
        else:
            # 득표자가 없어 처형 무산
            execute_event = GameEvent(
                day=self.day,
                phase=self.phase,
                event_type=EventType.EXECUTE,
                actor_id=-1,
                target_id=-1,
            )

        # 이벤트 기록
        if execute_event:
            self.history.append(execute_event)

            # 처형 성공 시에만 역할 공개 이벤트 추가
            if (
                execute_event.target_id != -1
                and self.players[execute_event.target_id].alive is False
            ):
                role_reveal_event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.SYSTEM_MESSAGE,
                    actor_id=-1,
                    target_id=execute_event.target_id,
                    value=self.players[execute_event.target_id].role,
                )
                self.history.append(role_reveal_event)

        return True

    def _process_night(self, actions: Dict[int, GameAction]) -> bool:
        """
        밤 단계 처리 - 마피아 투표 시스템 및 동점 시 무작위 선택 규칙 적용
        """

        night_targets = {}
        mafia_votes = []

        # --- 1. 마피아 투표 수집 및 다른 역할 행동 분리 ---
        other_role_actions = []
        for player_id, action in actions.items():
            p = self.players[player_id]
            if not p.alive:
                continue

            # 마피아는 투표를 수집합니다.
            if p.role == Role.MAFIA:
                if action.target_id != -1 and self.players[action.target_id].alive:
                    mafia_votes.append(action.target_id)
            # 마피아와 시민이 아닌 다른 직업은 나중에 처리하기 위해 따로 저장합니다.
            elif p.role != Role.CITIZEN:
                other_role_actions.append((player_id, action))

        # --- 2. 마피아 투표 집계 및 최종 타겟 결정 ---
        if mafia_votes:
            vote_counts = Counter(mafia_votes)
            max_votes = vote_counts.most_common(1)[0][1]
            top_targets = [p for p, v in vote_counts.items() if v == max_votes]

            # 동점이면 랜덤, 아니면 1명 선택
            mafia_target = random.choice(top_targets)
            night_targets[Role.MAFIA] = mafia_target

            # 단일 KILL 이벤트 생성 및 기록
            kill_event = GameEvent(
                day=self.day,
                phase=self.phase,
                event_type=EventType.KILL,
                actor_id=-1,  # 마피아 팀 전체의 행동으로 기록
                target_id=mafia_target,
            )
            self.history.append(kill_event)

        # --- 3. 의사, 경찰 등 다른 직업 행동 처리 및 기록 ---
        for player_id, action in other_role_actions:
            role = self.players[player_id].role
            target_id = action.target_id

            # 유효성 검사 (예: 경찰은 자신을 조사할 수 없음)
            if role == Role.POLICE and target_id == player_id:
                continue  # 유효하지 않은 행동은 무시

            # 대상이 유효한 생존자인 경우에만 처리
            if (
                target_id is not None
                and 0 <= target_id < len(self.players)
                and self.players[target_id].alive
            ):
                if role == Role.DOCTOR:
                    night_targets[Role.DOCTOR] = target_id
                    event = GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.PROTECT,
                        actor_id=player_id,
                        target_id=target_id,
                    )
                    self.history.append(event)

                elif role == Role.POLICE:
                    event = GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.POLICE_RESULT,
                        actor_id=player_id,
                        target_id=target_id,
                        value=self.players[target_id].role,
                    )
                    self.history.append(event)

        # --- 4. 최종 살해 및 보호 로직 ---
        final_mafia_target = night_targets.get(Role.MAFIA)
        doctor_target = night_targets.get(Role.DOCTOR)

        victim_id = -1

        if final_mafia_target is not None and final_mafia_target != doctor_target:
            self.players[final_mafia_target].alive = False
            victim_id = final_mafia_target

        ann_event = GameEvent(
            day=self.day + 1,
            phase=Phase.DAY_DISCUSSION,
            event_type=EventType.SYSTEM_MESSAGE,
            actor_id=-1,
            target_id=victim_id,
            value=None,
        )
        self.history.append(ann_event)
        return True

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
            if e.phase == Phase.GAME_START and e.event_type == EventType.SYSTEM_MESSAGE:
                if e.target_id == viewer_id:
                    filtered.append(e)
                continue
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

    def check_game_over(self, player_id: Optional[int] = None) -> Tuple[bool, bool]:
        """
        게임 종료 여부 및 승패 확인
        """
        m_count = sum(1 for p in self.players if p.role == Role.MAFIA and p.alive)
        c_count = sum(1 for p in self.players if p.role != Role.MAFIA and p.alive)

        termination_rules = [
            (self.day > config.game.MAX_DAYS, False),  # 1. 턴 초과 (마피아 승)
            (m_count == 0, True),  # 2. 마피아 전멸 (시민 승)
            (m_count >= c_count, False),  # 3. 마피아 과반 (마피아 승)
        ]

        for condition, citizen_win in termination_rules:
            if condition:
                if player_id is None:
                    return True, citizen_win

                is_mafia_team = self.players[player_id].role == Role.MAFIA
                my_win = citizen_win != is_mafia_team

                return True, my_win

        return False, False

    @property
    def winner(self) -> Optional[Role]:
        """
        게임 승리 팀을 반환합니다. 종료되지 않았으면 None 반환.
        """
        is_over, citizen_win = self.check_game_over()
        if not is_over:
            return None
        return Role.CITIZEN if citizen_win else Role.MAFIA
