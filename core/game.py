from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType
from state import GameStatus, GameEvent, PlayerStatus
from core.agent.llmAgent import LLMAgent
from core.agent.baseAgent import BaseAgent


class MafiaGame:
    def __init__(self, log_file=None, agents: Optional[List[BaseAgent]] = None, logger=None):
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
        self.log_file = log_file  # 레거시 지원
        self.logger = logger

    def reset(self, agents: Optional[List[BaseAgent]] = None) -> GameStatus:
        """
        Args:
            agents: \uc678\ubd80\uc5d0\uc11c \uc8fc\uc785\ubc1b\uc744 \uc5d0\uc774\uc804\ud2b8 \ub9ac\uc2a4\ud2b8
                    None\uc774\uba74 \uae30\ubcf8 LLMAgent 8\uba85\uc73c\ub85c \ucd08\uae30\ud654
        """
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.history = []
        self._last_votes = []  # \ud22c\ud45c \uacb0\uacfc \ucd08\uae30\ud654
        
        # \uc5d0\uc774\uc804\ud2b8 \ucd08\uae30\ud654: \uc678\ubd80 \uc8fc\uc785 \ub610\ub294 \uae30\ubcf8 LLMAgent \uc0dd\uc131
        if agents is not None:
            self.players = agents
        elif not self.players:  # \uc0dd\uc131\uc790\uc5d0\uc11c\ub3c4 \ubc1b\uc9c0 \uc54a\uc558\uc73c\uba74 \uae30\ubcf8 \uc0dd\uc131
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
                    phase=Phase.DAY_DISCUSSION,
                    event_type=EventType.POLICE_RESULT,  # 역할 공개 이벤트로 활용
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

        if self.phase == Phase.DAY_DISCUSSION:
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
        for _ in range(2):
            ended = False
            for p in self.players:
                if not p.alive:
                    continue
                
                try:
                    p.observe(self.get_game_status(p.id))
                    action_dict = p.get_action()  # Dict[str, Any] 반환
                    print(action_dict, end="\n")
                    
                    if "error" in action_dict:
                        continue
                    
                    if action_dict.get("discussion_status") == "End":
                        ended = True
                        break

                    role_id = action_dict.get("role")
                    target_id = action_dict.get("target_id")
                    reason = action_dict.get("reason", "")

                    # 이벤트 객체 생성
                    event = GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.CLAIM,
                        actor_id=p.id,
                        target_id=target_id,
                        value=Role(role_id) if role_id is not None else None,
                    )
                    
                    # 이벤트를 history와 logger에 기록
                    self.history.append(event)
                    if self.logger:
                        self.logger.log_event(event)
                except Exception as e:
                    continue
            
            if ended:
                break

        # Phase End: observe만 호출 (내부에서 update_belief 수행)
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_vote(self):
        votes = [0] * len(self.players)
        for p in self.players:
            if not p.alive:
                continue
            p.observe(self.get_game_status(p.id))
            
            action_dict = p.get_action()  # Dict[str, Any] 반환
            
            if "error" in action_dict:
                continue
            
            target = action_dict.get("target_id")
            reason = action_dict.get("reason", "")

            if target == p.id:
                target = -1

            if target is not None and target != -1:
                votes[target] += 1
                
                # 이벤트 생성 및 로깅
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=EventType.VOTE,
                    actor_id=p.id,
                    target_id=target,
                    value=None,
                )
                self.history.append(event)
                if self.logger:
                    self.logger.log_event(event)
        
        # 투표 결과 저장 (다음 단계에서 사용)
        self._last_votes = votes

        # Phase End: observe만 호출
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_execute(self):
        if not self._last_votes:
            return
        
        max_v = max(self._last_votes)
        if max_v > 0:
            targets = [i for i, v in enumerate(self._last_votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]
                final_score = 0
                
                # 각 플레이어의 처형 동의 여부 확인
                for p in self.players:
                    if not p.alive:
                        continue
                    p.observe(self.get_game_status(p.id))
                    
                    action_dict = p.get_action()  # Dict[str, Any] 반환
                    
                    if "error" in action_dict:
                        continue
                    
                    agree = action_dict.get("agree_execution", 0)
                    final_score += agree
                
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

        # Phase End: observe만 호출
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_night(self):
        """
        밤 단계 처리 - 역할별 로직을 매핑으로 분리하여 선언적 구조 유지
        """
        # 1. 역할별 행동 설정 매핑
        role_config = {
            Role.MAFIA: {"event_type": EventType.KILL, "value": None},
            Role.DOCTOR: {"event_type": EventType.PROTECT, "value": None},
            Role.POLICE: {"event_type": EventType.POLICE_RESULT, "value": "role_of_target"},
        }
        
        # 2. 밤 행동 결과 저장
        night_targets = {}
        
        for p in self.players:
            if not p.alive or p.role == Role.CITIZEN:
                continue
            
            p.observe(self.get_game_status(p.id))
            action_dict = p.get_action()  # Dict[str, Any] 반환
            
            if "error" in action_dict:
                continue
            
            target_id = action_dict.get("target_id")
            if target_id is not None and self.players[target_id].alive:
                # 3. 역할별 설정 가져오기
                config = role_config.get(p.role)
                if not config:
                    continue
                
                # 4. 타겟 저장 (살해/보호 판정용)
                night_targets[p.role] = target_id
                
                # 5. 이벤트 생성
                event_value = None
                if config["value"] == "role_of_target":
                    event_value = self.players[target_id].role
                
                event = GameEvent(
                    day=self.day,
                    phase=self.phase,
                    event_type=config["event_type"],
                    actor_id=p.id,
                    target_id=target_id,
                    value=event_value,
                )
                self.history.append(event)
                if self.logger:
                    self.logger.log_event(event)

        # 7. 살해 및 보호 처리
        mafia_target = night_targets.get(Role.MAFIA)
        doctor_target = night_targets.get(Role.DOCTOR)
        
        if mafia_target is not None and mafia_target != doctor_target:
            self.players[mafia_target].alive = False

        # Phase End: observe만 호출
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
