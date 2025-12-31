from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType
from state import GameStatus, GameEvent, PlayerStatus
from core.agent.llmAgent import LLMAgent
from core.agent.baseAgent import BaseAgent


class MafiaGame:
    def __init__(self, log_file=None, agents: Optional[List[BaseAgent]] = None):
        """
        Args:
            log_file: \ub85c\uadf8 \ud30c\uc77c \ud578\ub4e4
            agents: \uc678\ubd80\uc5d0\uc11c \uc8fc\uc785\ubc1b\uc744 \uc5d0\uc774\uc804\ud2b8 \ub9ac\uc2a4\ud2b8 (\uc120\ud0dd\uc801)
        """
        self.players: List[BaseAgent] = agents if agents is not None else []
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.history: List[GameEvent] = []
        self.log_file = log_file

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

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
            self._log(f"\ud50c\ub808\uc774\uc5b4 {p.id}: {p.role.name}")

        return self.get_game_status()

    def process_turn(self) -> Tuple[GameStatus, bool, bool]:
        """
        게임 턴 진행 - 외부에서 액션을 주입받지 않고 에이전트에게 직접 의사를 물음
        """
        is_over, is_win = self.check_game_over()
        if is_over:
            return self.get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day} | {self.phase.name}]")

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
        self._log("토론 시작")
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
                        self._log(f"플레이어 {p.id} 액션 오류: {action_dict['error']}")
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
                    
                    # LogManager를 통한 내러티브 생성 (직접 문자열 조립 제거)
                    if self.logger:
                        narrative = self.logger.interpret_event(event)
                        log_message = narrative + (f" 이유: {reason}" if reason else "")
                    else:
                        # Fallback: logger가 없을 경우 기본 포맷
                        role = p.role.name if role_id is None else Role(role_id).name
                        if role_id is not None:
                            if target_id == p.id or target_id == -1:
                                log_message = f"Player {p.id}는 자신이 {role}라고 주장"
                            else:
                                log_message = f"Player {p.id}는 Player {target_id}가 {role}라고 주장"
                        else:
                            log_message = f"Player {p.id}가 침묵."
                        log_message += (f" 이유: {reason}" if reason else "")
                    
                    self._log(log_message)
                    self.history.append(event)
                except Exception as e:
                    self._log(f"플레이어 {p.id} 액션 처리 중 에러 발생: {e}")
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
                self._log(f"플레이어 {p.id} 투표 오류: {action_dict['error']}")
                continue
            
            target = action_dict.get("target_id")
            reason = action_dict.get("reason", "")

            if target == p.id:
                self._log(
                    f"플레이어 {p.id}는 자신에게 투표할 수 없습니다. 투표 무효 처리."
                )
                target = -1
            else:
                self._log(f"플레이어 {p.id} 투표: {target}. 이유: {reason}")

            if target is not None and target != -1:
                votes[target] += 1
                self.history.append(
                    GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.VOTE,
                        actor_id=p.id,
                        target_id=target,
                        value=None,
                    )
                )
        
        # 투표 결과 저장 (다음 단계에서 사용)
        self._last_votes = votes

        # Phase End: observe만 호출
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_execute(self):
        if not self._last_votes:
            self._log("투표 결과가 없습니다.")
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
                        self._log(f"플레이어 {p.id} 처형 동의 오류: {action_dict['error']}")
                        continue
                    
                    agree = action_dict.get("agree_execution", 0)
                    final_score += agree
                    self._log(f"플레이어 {p.id} 처형 동의: {agree}")
                
                success = final_score > 0
                if success:
                    self.players[target_id].alive = False
                    self._log(
                        f"처형 성공: {target_id} ({self.players[target_id].role.name})"
                    )
                    self.history.append(
                        GameEvent(
                            day=self.day,
                            phase=self.phase,
                            event_type=EventType.POLICE_RESULT,
                            actor_id=-1,
                            target_id=target_id,
                            value=self.players[target_id].role,
                        )
                    )
                self.history.append(
                    GameEvent(
                        day=self.day,
                        phase=self.phase,
                        event_type=EventType.EXECUTE,
                        actor_id=-1,
                        target_id=target_id,
                        value=success,
                    )
                )

        # Phase End: observe만 호출
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))

    def _process_night(self):
        m_target, d_target, p_target = None, None, None
        for p in self.players:
            if not p.alive or p.role == Role.CITIZEN:
                continue
            p.observe(self.get_game_status(p.id))
            
            action_dict = p.get_action()  # Dict[str, Any] 반환
            
            if "error" in action_dict:
                self._log(f"플레이어 {p.id} 야간 행동 오류: {action_dict['error']}")
                continue
            
            target = action_dict.get("target_id")
            if target is not None and self.players[target].alive:
                if p.role == Role.MAFIA:
                    m_target = target
                    self._log(f"마피아 {p.id}의 살해 목표: {m_target}")
                    self.history.append(
                        GameEvent(
                            day=self.day,
                            phase=self.phase,
                            event_type=EventType.KILL,
                            actor_id=p.id,
                            target_id=target,
                        )
                    )
                elif p.role == Role.DOCTOR:
                    d_target = target
                    self._log(f"의사 {p.id}의 보호 목표: {d_target}")
                    self.history.append(
                        GameEvent(
                            day=self.day,
                            phase=self.phase,
                            event_type=EventType.PROTECT,
                            actor_id=p.id,
                            target_id=target,
                        )
                    )
                elif p.role == Role.POLICE:
                    p_target = target
                    self._log(f"경찰 {p.id}의 조사 목표: {p_target}")
                    self.history.append(
                        GameEvent(
                            day=self.day,
                            phase=self.phase,
                            event_type=EventType.POLICE_RESULT,
                            actor_id=p.id,
                            target_id=target,
                            value=self.players[target].role,
                        )
                    )

        if m_target is not None and m_target != d_target:
            self.players[m_target].alive = False
            self._log(f"살해 발생: {m_target}")

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
