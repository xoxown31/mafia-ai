from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType, ActionType
from state import GameStatus, GameEvent, PlayerStatus, MafiaAction
from core.agent.llmAgent import LLMAgent
from core.agent.baseAgent import BaseAgent


class MafiaGame:
    def __init__(self, agents: List[BaseAgent], logger=None):
        """
        Args:
            agents: 외부에서 주입받을 에이전트 리스트 (필수)
            logger: LogManager 인스턴스 (선택적)
        """
        self.players: List[BaseAgent] = agents
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.discussion_round = 0
        self.history: List[GameEvent] = []
        self.logger = logger

    def reset(self) -> GameStatus:
        """
        게임 상태 초기화
        """
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.discussion_round = 0
        self.history = []
        self._last_votes = []  # 투표 결과 초기화
        
        # 에이전트 상태 초기화 (필요 시)
        # 외부에서 주입된 에이전트를 그대로 사용

        roles = config.game.DEFAULT_ROLES.copy()
        random.shuffle(roles)
        for p, r in zip(self.players, roles):
            p.role = r
            # 역할 할당 이벤트 로깅
            if self.logger:
                # GameEvent로 기록 (JSONL 저장용)
                event = GameEvent(
                    day=0,
                    phase=Phase.DAY_DISCUSSION,
                    event_type=EventType.POLICE_RESULT,  # 역할 공개 이벤트로 활용
                    actor_id=-1,
                    target_id=p.id,
                    value=p.role,
                )
                self.logger.log_event(event)

        return self.get_game_status()

    def step_phase(self, actions: Dict[int, MafiaAction]) -> Tuple[GameStatus, bool, bool]:
        """
        게임 페이즈 진행 - 외부에서 주입된 액션 처리
        단일 페이즈(Phase) 단위로 실행을 멈추고 제어권을 반환
        """
        is_over, is_win = self.check_game_over()
        if is_over:
            return self.get_game_status(), is_over, is_win

        if self.phase == Phase.DAY_DISCUSSION:
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
        return self.get_game_status(), is_over, is_win

    def _process_discussion(self, actions: Dict[int, MafiaAction]) -> bool:
        """
        토론 단계 처리 - 외부에서 주입된 액션 처리
        Returns:
            bool: 페이즈 종료 여부 (True면 다음 페이즈로)
        """
        alive_count = sum(1 for p in self.players if p.alive)
        
        # 순차적 반영: 8명의 액션을 루프를 돌며 하나씩 적용
        # actions 딕셔너리는 {player_id: action} 형태
        # 순서는 player_id 순으로 하거나 랜덤으로 할 수 있음. 여기서는 ID 순으로 처리.
        sorted_actions = sorted(actions.items(), key=lambda x: x[0])
        
        pass_count = 0
        
        for player_id, action in sorted_actions:
            player = self.players[player_id]
            if not player.alive:
                continue
                
            # PASS 카운트
            if action.target_id == -1 and action.claim_role is None:
                pass_count += 1
            
            # CLAIM 이벤트 생성 및 기록
            if action.claim_role is not None:
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.CLAIM,
                    actor_id=player_id,
                    target_id=action.target_id if action.target_id != -1 else None,
                    value=action.claim_role
                )
                self.history.append(event)
                if self.logger:
                    self.logger.log_event(event)
                    
                # 에이전트들에게 즉시 알림 (순차적 반영의 핵심)
                # 하지만 현재 구조상 step_phase가 끝나야 env.step이 반환되고, 
                # 그제서야 에이전트들이 다음 observation을 받음.
                # RL 에이전트가 "이벤트 시퀀스를 하나씩 훑으며" 학습하려면,
                # 여기서 발생한 이벤트를 모아서 env가 반환할 때 시퀀스로 줘야 함.
                # 하지만 현재 env 구조는 step마다 하나의 observation만 반환함.
                # 따라서, env.step() 내에서 여러 번의 model forward가 일어나지 않음.
                # 
                # 요구사항: "각 액션이 적용될 때마다 발생하는 GameEvent를 리스트로 수집하여, 
                # RL 에이전트가 이 시퀀스를 하나씩 훑으며 RNN의 Hidden State를 갱신할 수 있게 한다."
                # 
                # 이를 위해서는 env.step()이 반환하는 observation에 'events' 리스트가 포함되어야 하고,
                # RL 에이전트(PPO)는 이 리스트를 순회하며 hidden state를 업데이트해야 함.
                # 현재 env._encode_observation은 가장 최근 이벤트 하나만 반영함.
                # 
                # 해결책:
                # 1. GameStatus에 'new_events' 필드를 추가하여 이번 step에서 발생한 모든 이벤트를 담음.
                # 2. env._encode_observation에서 이 'new_events'를 인코딩하여 반환?
                #    -> 78차원 벡터는 단일 시점의 상태임.
                #    -> RNN은 시퀀스 입력을 받을 수 있음.
                #    -> env.step()에서 반환하는 observation을 [seq_len, 78] 형태로 만들어서 반환하면 됨.
                #    -> 즉, 이번 턴에 발생한 이벤트 N개에 대해 각각 78차원 벡터를 생성하여 리스트로 반환.
                
        # 모든 액션 처리 후 종료 조건 확인
        if pass_count >= alive_count * 0.5:  # 과반수 이상 PASS 시 투표로 넘어감
            return True
            
        self.discussion_round += 1
        if self.discussion_round >= config.game.MAX_DISCUSSION_ROUNDS:
            return True
            
        return False

    def _process_vote(self, actions: Dict[int, MafiaAction]):
        """
        투표 단계 처리 - 외부에서 주입된 액션 처리
        """
        votes = [0] * len(self.players)
        
        # Phase 2: 모든 투표를 한 번에 처리
        for player_id, action in actions.items():
            target_id = action.target_id
            
            # 유효한 투표인지 확인
            if target_id != -1 and 0 <= target_id < len(self.players) and self.players[target_id].alive:
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
        
        return True

    def _process_execute(self, actions: Dict[int, MafiaAction]) -> bool:
        """
        처형 단계 처리 - 외부에서 주입된 액션 처리
        """
        if not self._last_votes:
            return True
        
        max_v = max(self._last_votes)
        if max_v > 0:
            targets = [i for i, v in enumerate(self._last_votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]
                
                # Phase 2: 모든 동의를 한 번에 처리
                final_score = 0
                for player_id, action in actions.items():
                    # 처형 동의 여부 확인 (임시: dict 호환성 유지)
                    if isinstance(action, dict):
                        agree = action.get("agree_execution", 0)
                        final_score += agree
                    elif isinstance(action, MafiaAction):
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
                        event_type=EventType.POLICE_RESULT,
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
        
        return True

    def _process_night(self, actions: Dict[int, MafiaAction]) -> bool:
        """
        밤 단계 처리 - 외부에서 주입된 액션 처리
        """
        # 1. 역할별 행동 설정 매핑
        role_config = {
            Role.MAFIA: {"event_type": EventType.KILL, "value": None},
            Role.DOCTOR: {"event_type": EventType.PROTECT, "value": None},
            Role.POLICE: {"event_type": EventType.POLICE_RESULT, "value": "role_of_target"},
        }
        
        # Phase 2: 모든 밤 행동을 한 번에 처리
        night_targets = {}
        for player_id, action in actions.items():
            p = self.players[player_id]
            if not p.alive or p.role == Role.CITIZEN:
                continue
            
            target_id = action.target_id
            
            # 유효한 타겟인지 확인
            if target_id != -1 and 0 <= target_id < len(self.players) and self.players[target_id].alive:
                # 역할별 설정 가져오기
                config = role_config.get(p.role)
                if not config:
                    continue
                
                # 타겟 저장 (살해/보호 판정용)
                night_targets[p.role] = target_id
                
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
