from typing import List, Dict, Tuple, Optional
import random
import json
from config import config, Role, Phase, EventType
from state import GameStatus, GameEvent, PlayerStatus
from core.agent.llmAgent import LLMAgent

class MafiaGame:
    def __init__(self, log_file=None):
        self.players: List[LLMAgent] = []
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.history: List[GameEvent] = []
        self.log_file = log_file

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> GameStatus:
        self.day = 1
        self.phase = Phase.DAY_DISCUSSION
        self.history = []
        self.players = [LLMAgent(player_id=i) for i in range(config.game.PLAYER_COUNT)]
        
        roles = config.game.DEFAULT_ROLES.copy()
        random.shuffle(roles)
        for p, r in zip(self.players, roles):
            p.role = r
            self._log(f"플레이어 {p.id}: {p.role.name}")
        
        return self.get_game_status()

    def process_turn(self, action: int = -1) -> Tuple[GameStatus, bool, bool]:
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
        for _ in range(2):
            ended = False
            for p in self.players:
                if not p.alive: continue
                p.observe(self.get_game_status(p.id))
                resp = p.get_action()
                try:
                    data = json.loads(resp)
                    if data.get("discussion_status") == "End":
                        ended = True
                        break
                    
                    role_id = data.get("role")
                    target_id = data.get("target_id")
                    self._log(f"플레이어 {p.id}: {data.get('reason', '')}")
                    
                    self.history.append(GameEvent(
                        day=self.day, phase=self.phase, event_type=EventType.CLAIM,
                        actor_id=p.id, target_id=target_id, 
                        value=Role(role_id) if role_id is not None else None
                    ))
                except: continue
            if ended: break
        
        # Phase End: Update Beliefs
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))
                p.update_belief(self.history)

    def _process_vote(self):
        votes = [0] * len(self.players)
        for p in self.players:
            if not p.alive: continue
            p.observe(self.get_game_status(p.id))
            try:
                data = json.loads(p.get_action())
                target = data.get("target_id")
                if target is not None and target != -1:
                    votes[target] += 1
                    self.history.append(GameEvent(
                        day=self.day, phase=self.phase, event_type=EventType.VOTE,
                        actor_id=p.id, target_id=target, value=None
                    ))
            except: continue
        self._last_votes = votes

        # Phase End: Update Beliefs
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))
                p.update_belief(self.history)

    def _process_execute(self):
        max_v = max(self._last_votes)
        if max_v > 0:
            targets = [i for i, v in enumerate(self._last_votes) if v == max_v]
            if len(targets) == 1:
                target_id = targets[0]
                final_score = 0
                for p in self.players:
                    if not p.alive: continue
                    p.observe(self.get_game_status(p.id))
                    try:
                        data = json.loads(p.get_action())
                        final_score += data.get("agree_execution", 0)
                    except: continue
                
                success = final_score > 0
                if success:
                    self.players[target_id].alive = False
                    self._log(f"처형 성공: {target_id} ({self.players[target_id].role.name})")
                    self.history.append(GameEvent(
                        day=self.day, phase=self.phase, event_type=EventType.POLICE_RESULT,
                        actor_id=-1, target_id=target_id, value=self.players[target_id].role
                    ))
                self.history.append(GameEvent(
                    day=self.day, phase=self.phase, event_type=EventType.EXECUTE,
                    actor_id=-1, target_id=target_id, value=success
                ))

        # Phase End: Update Beliefs
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))
                p.update_belief(self.history)

    def _process_night(self):
        m_target, d_target, p_target = None, None, None
        for p in self.players:
            if not p.alive or p.role == Role.CITIZEN: continue
            p.observe(self.get_game_status(p.id))
            try:
                data = json.loads(p.get_action())
                target = data.get("target_id")
                if target is not None and self.players[target].alive:
                    if p.role == Role.MAFIA: 
                        m_target = target
                        self.history.append(GameEvent(day=self.day, phase=self.phase, event_type=EventType.KILL, actor_id=p.id, target_id=target))
                    elif p.role == Role.DOCTOR: 
                        d_target = target
                        self.history.append(GameEvent(day=self.day, phase=self.phase, event_type=EventType.PROTECT, actor_id=p.id, target_id=target))
                    elif p.role == Role.POLICE: 
                        p_target = target
                        self.history.append(GameEvent(day=self.day, phase=self.phase, event_type=EventType.POLICE_RESULT, actor_id=p.id, target_id=target, value=self.players[target].role))
            except: continue

        if m_target is not None and m_target != d_target:
            self.players[m_target].alive = False
            self._log(f"살해 발생: {m_target}")

        # Phase End: Update Beliefs
        for p in self.players:
            if p.alive:
                p.observe(self.get_game_status(p.id))
                p.update_belief(self.history)

    def get_game_status(self, viewer_id: Optional[int] = None) -> GameStatus:
        if viewer_id is None:
            return GameStatus(day=self.day, phase=self.phase, my_id=-1, my_role=Role.CITIZEN,
                              players=[PlayerStatus(id=p.id, alive=p.alive) for p in self.players],
                              action_history=self.history)
        
        viewer = self.players[viewer_id]
        filtered = []
        if viewer.role == Role.MAFIA:
            for p in self.players:
                if p.role == Role.MAFIA and p.id != viewer_id:
                    filtered.append(GameEvent(day=0, phase=Phase.DAY_DISCUSSION, event_type=EventType.POLICE_RESULT, actor_id=-1, target_id=p.id, value=Role.MAFIA))
        
        for e in self.history:
            if e.phase == Phase.NIGHT:
                if e.actor_id == viewer_id or (viewer.role == Role.MAFIA and self.players[e.actor_id].role == Role.MAFIA) or (e.event_type == EventType.POLICE_RESULT and e.actor_id == viewer_id):
                    filtered.append(e)
            else: filtered.append(e)
            
        return GameStatus(day=self.day, phase=self.phase, my_id=viewer_id, my_role=viewer.role,
                          players=[PlayerStatus(id=p.id, alive=p.alive) for p in self.players],
                          action_history=filtered)

    def check_game_over(self) -> Tuple[bool, bool]:
        m_count = sum(1 for p in self.players if p.role == Role.MAFIA and p.alive)
        c_count = sum(1 for p in self.players if p.role != Role.MAFIA and p.alive)
        if self.day > config.game.MAX_DAYS: return True, False
        if m_count == 0: return True, True
        if m_count >= c_count: return True, False
        return False, False