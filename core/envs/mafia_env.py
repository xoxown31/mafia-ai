import functools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from core.engine.game import MafiaGame
from core.agents.base_agent import BaseAgent
from config import config, Role, Phase, EventType, ActionType
from core.engine.state import GameStatus, GameAction, PlayerStatus, GameEvent


class EnvAgent(BaseAgent):
    """
    Environment internal agent placeholder to satisfy MafiaGame requirements.
    This agent does not perform any logic; it just holds state.
    """
    def get_action(self, status: GameStatus) -> GameAction:
        return GameAction(target_id=-1, claim_role=None)


class MafiaEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "mafia_v1"}

    def __init__(self, render_mode=None, logger=None):
        self.possible_agents = [f"player_{i}" for i in range(config.game.PLAYER_COUNT)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        self.logger = logger

        # Create dummy agents for the engine
        # MafiaGame expects a list of BaseAgent instances
        self.internal_agents = [EnvAgent(i) for i in range(config.game.PLAYER_COUNT)]
        self.game = MafiaGame(agents=self.internal_agents, logger=logger)

        # === [Multi-Discrete Action Space] ===
        # 형태: [Target, Role]
        # - Target: 0=None, 1~8=Player 0~7 (9개)
        # - Role: 0=None, 1~4=Role Enum (5개)
        self.action_spaces = {
            agent: spaces.MultiDiscrete(config.game.ACTION_DIMS)
            for agent in self.possible_agents
        }

        # Observation Space: 78차원 슬림화
        # 자기 정보(12) + 게임 상황(4) + 주관적 신뢰도(32) + 직전 사건(30)
        obs_dim = config.game.OBS_DIM
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-1, high=1, shape=(obs_dim,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(14,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }

        # 이전 턴의 투표 기록 저장 (삭제됨)
        # self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # PettingZoo API: reset returns (observations, infos)
        self.agents = self.possible_agents[:]
        self.game.reset()
        # self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)

        observations = {
            agent: self._encode_observation(self._agent_to_id(agent))
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        # Convert string keys to int keys for the engine
        engine_actions = {}
        for agent_id, action in actions.items():
            pid = self._agent_to_id(agent_id)
            if isinstance(action, (list, np.ndarray)):
                engine_actions[pid] = GameAction.from_multi_discrete(action)
            elif isinstance(action, GameAction):
                engine_actions[pid] = action
            elif isinstance(action, dict):
                engine_actions[pid] = action
            else:
                # Fallback or error
                pass

        # 턴 진행 전 상태 저장 (보상 계산용)
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        # 게임 진행
        status, is_over, is_win = self.game.step_phase(engine_actions)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            pid = self._agent_to_id(agent)
            
            observations[agent] = {
                "observation": self._encode_observation(pid),
                "action_mask": self._get_action_mask(pid),
            }

            if is_over:
                _, my_win = self.game.check_game_over(player_id=pid)
            else:
                my_win = False
            
            rewards[agent] = self._calculate_reward(pid, prev_alive, prev_phase, engine_actions.get(pid), is_over, is_win)
            terminations[agent] = is_over
            truncations[agent] = False
            infos[agent] = {"day": status.day, "phase": status.phase, "win": my_win}

        if is_over:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """게임 상태 렌더링"""
        phase_names = {
            Phase.DAY_DISCUSSION: "Discussion",
            Phase.DAY_VOTE: "Vote",
            Phase.DAY_EXECUTE: "Execute",
            Phase.NIGHT: "Night",
        }
        phase_str = phase_names.get(self.game.phase, str(self.game.phase))
        status = self.game.get_game_status()
        alive_indices = [p.id for p in status.players if p.alive]
        print(f"[Day {self.game.day}] {phase_str} | Alive: {alive_indices}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _agent_to_id(self, agent_str):
        return int(agent_str.split("_")[1])

    def _id_to_agent(self, agent_id):
        return f"player_{agent_id}"

    # === Helper Methods (Copied and adapted from previous implementation) ===

    def _calculate_reward(
        self, agent_id, prev_alive, prev_phase, mafia_action, done, win
    ):
        reward = 0.0
        agent = self.game.players[agent_id]
        role = agent.role

        # 1. 승패 보상 (Sparse Reward)
        if done:
            is_mafia_team = role == Role.MAFIA
            my_win = (win and not is_mafia_team) or (not win and is_mafia_team)

            if my_win:
                reward += 10.0
            else:
                reward -= 5.0

        # 2. 생존 보상 (약하게)
        if not agent.alive:
            reward -= 0.1
        else:
            reward += 0.05

        # 3. 역할 기반 행동 보상
        action_target = -1
        if hasattr(mafia_action, "target_id"):
            action_target = mafia_action.target_id

        if role == Role.CITIZEN:
            reward += self._calculate_citizen_reward(action_target, prev_phase)
        elif role == Role.MAFIA:
            reward += self._calculate_mafia_reward(action_target, prev_phase)
        elif role == Role.POLICE:
            reward += self._calculate_police_reward(action_target, prev_phase)
        elif role == Role.DOCTOR:
            reward += self._calculate_doctor_reward(prev_alive, action_target, prev_phase)
            
        # 4. 기만 및 설득 보상 (New)
        reward += self._calculate_deception_reward(agent_id, role, prev_phase)
        reward += self._calculate_persuasion_reward(agent_id, role, prev_phase, mafia_action)
        
        return reward

    def _calculate_deception_reward(self, agent_id, role, phase):
        """
        마피아 기만 보상:
        - 낮 토론 단계에서 타인이 나를 경찰/의사로 지목 시 보상
        - 투표 단계에서 나의 득표수가 생존자 평균보다 낮을 경우 보상
        """
        if role != Role.MAFIA:
            return 0.0
            
        reward = 0.0
        
        # 1. 낮 토론 단계: 타인이 나를 경찰/의사로 지목
        if phase == Phase.DAY_DISCUSSION:
            if self.game.history:
                last_evt = self.game.history[-1]
                if (last_evt.event_type == EventType.CLAIM and 
                    last_evt.target_id == agent_id and 
                    last_evt.actor_id != agent_id):
                    
                    if last_evt.value in [Role.POLICE, Role.DOCTOR]:
                        reward += 0.5  # 기만 성공
        
        # 2. 투표 단계: 득표수가 평균보다 낮음
        elif phase == Phase.DAY_VOTE:
            alive_players = [p for p in self.game.players if p.alive]
            if not alive_players: return 0.0
            
            avg_votes = sum(p.vote_count for p in alive_players) / len(alive_players)
            my_votes = self.game.players[agent_id].vote_count
            
            if my_votes < avg_votes:
                reward += 0.2
                
        return reward

    def _calculate_persuasion_reward(self, agent_id, role, phase, my_action):
        """
        시민 설득 보상:
        - 내가 지목한 타겟에게 다른 플레이어들이 동조하여 투표하거나 지목할 경우 보상
        """
        # 마피아도 설득이 필요하지만, 요구사항에 따라 시민 팀에게만 적용하거나
        # 혹은 전체에게 적용할 수도 있음. 여기서는 시민 팀(시민, 경찰, 의사)에게 적용.
        if role == Role.MAFIA:
            return 0.0
            
        reward = 0.0
        
        if not my_action or my_action.target_id == -1:
            return 0.0
            
        target_id = my_action.target_id
        
        if phase == Phase.DAY_VOTE:
            # 타겟의 득표수 확인 (나 제외)
            target_votes = self.game.players[target_id].vote_count
            if target_votes > 1: # 나 말고 최소 1명 더
                reward += 0.1 * (target_votes - 1)
                
        elif phase == Phase.DAY_DISCUSSION:
            # 최근 history 확인
            count = 0
            recent_events = self.game.history[-config.game.PLAYER_COUNT:]
            for evt in recent_events:
                if (evt.event_type == EventType.CLAIM and 
                    evt.target_id == target_id and 
                    evt.actor_id != agent_id):
                    count += 1
            
            if count > 0:
                reward += 0.2 * count
                
        return reward

    def _calculate_citizen_reward(self, action, phase):
        if action == -1:
            return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if phase == Phase.DAY_VOTE:
                if target.role == Role.MAFIA:
                    reward += 15.0
                    if not target.alive:
                        reward += 10.0
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 8.0
                elif target.role == Role.CITIZEN:
                    reward -= 2.0
            elif phase == Phase.DAY_DISCUSSION:
                if target.role == Role.MAFIA:
                    reward += 3.0
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 1.0
        return reward

    def _calculate_mafia_reward(self, action, phase):
        if action == -1:
            return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if phase == Phase.DAY_VOTE:
                if target.role == Role.POLICE:
                    reward += 20.0
                    if not target.alive:
                        reward += 15.0
                elif target.role == Role.DOCTOR:
                    reward += 15.0
                    if not target.alive:
                        reward += 10.0
                elif target.role == Role.CITIZEN:
                    reward += 5.0
                    if not target.alive:
                        reward += 3.0
                elif target.role == Role.MAFIA:
                    reward -= 25.0
            elif phase == Phase.NIGHT:
                if target.role == Role.POLICE:
                    reward += 25.0
                    if not target.alive:
                        reward += 15.0
                elif target.role == Role.DOCTOR:
                    reward += 18.0
                    if not target.alive:
                        reward += 12.0
                elif target.role == Role.CITIZEN:
                    reward += 8.0
                    if not target.alive:
                        reward += 5.0
        return reward

    def _calculate_police_reward(self, action, phase):
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
        reward = 0.0
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if target.role == Role.MAFIA:
                reward += 20.0
            else:
                reward += 2.0
        return reward

    def _calculate_doctor_reward(self, prev_alive, action, phase):
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
        reward = 0.0
        current_alive_count = sum(p.alive for p in self.game.players)
        prev_alive_count = sum(prev_alive)
        if current_alive_count == prev_alive_count:
            reward += 25.0
            if 0 <= action < len(self.game.players):
                target = self.game.players[action]
                if target.role == Role.POLICE:
                    reward += 15.0
                elif target.role == Role.DOCTOR:
                    reward += 10.0
                elif target.role == Role.CITIZEN:
                    reward += 5.0
        else:
            reward += 1.0
        return reward

    def _encode_observation(
        self, agent_id: int, target_event: Optional[GameEvent] = None
    ) -> np.ndarray:
        """
        46차원 관측 벡터 생성 (Belief Matrix 제거)
        target_event가 주어지면 해당 이벤트를 '직전 사건'으로 인코딩.
        없으면 가장 최근 이벤트를 사용.
        """
        status = self.game.get_game_status(agent_id)
        agent = self.game.players[agent_id]

        # 1. 자기 정보 (12)
        # ID One-hot (8)
        id_vec = np.zeros(8, dtype=np.float32)
        id_vec[agent_id] = 1.0

        # Role One-hot (4)
        role_vec = np.zeros(4, dtype=np.float32)
        role_vec[int(status.my_role)] = 1.0

        # 2. 게임 상황 (4)
        # Day (1) - 정규화 (최대 15일 가정)
        day_vec = np.array([status.day / 15.0], dtype=np.float32)

        # Phase One-hot (3)
        phase_vec = np.zeros(3, dtype=np.float32)
        if status.phase == Phase.DAY_DISCUSSION:
            phase_vec[0] = 1.0
        elif status.phase == Phase.DAY_VOTE:
            phase_vec[1] = 1.0
        else:  # Execute or Night
            phase_vec[2] = 1.0
            
        # 3. 직전 사건 (30)
        # target_event가 있으면 그것을 사용, 없으면 history의 마지막 사용
        last_event = target_event
        if last_event is None and status.action_history:
            last_event = status.action_history[-1]

        if last_event:
            # Actor ID (9): Player 0~7 + System(-1) -> Index 8
            actor_vec = np.zeros(9, dtype=np.float32)
            if last_event.actor_id == -1:
                actor_vec[8] = 1.0
            else:
                actor_vec[last_event.actor_id] = 1.0

            # Target ID (9): Player 0~7 + None -> Index 8
            target_vec = np.zeros(9, dtype=np.float32)
            if last_event.target_id is None or last_event.target_id == -1:
                target_vec[8] = 1.0
            else:
                target_vec[last_event.target_id] = 1.0

            # Value/Role (5): Role 0~3 + None -> Index 4
            value_vec = np.zeros(5, dtype=np.float32)
            if last_event.value is None:
                value_vec[4] = 1.0
            elif isinstance(last_event.value, Role):
                value_vec[int(last_event.value)] = 1.0
            else:
                value_vec[4] = 1.0

            # Event Type (7)
            type_vec = np.zeros(7, dtype=np.float32)
            try:
                t_idx = int(last_event.event_type) - 1
                if 0 <= t_idx < 7:
                    type_vec[t_idx] = 1.0
            except:
                pass

        else:
            # 이벤트 없음 (초기 상태)
            actor_vec = np.zeros(9, dtype=np.float32)
            actor_vec[8] = 1.0  # System
            target_vec = np.zeros(9, dtype=np.float32)
            target_vec[8] = 1.0  # None
            value_vec = np.zeros(5, dtype=np.float32)
            value_vec[4] = 1.0  # None
            type_vec = np.zeros(7, dtype=np.float32)  # None type?

        # Concatenate all
        obs = np.concatenate([
            id_vec,      # 8
            role_vec,    # 4
            day_vec,     # 1
            phase_vec,   # 3
            actor_vec,   # 9
            target_vec,  # 9
            value_vec,   # 5
            type_vec     # 7
        ]) # Total 46
        
        return obs

    def _get_action_mask(self, agent_id):
        # 기존 로직 활용 또는 단순화
        # Target(9) + Role(5) = 14
        mask = np.ones(14, dtype=np.int8)

        status = self.game.get_game_status(agent_id)
        agent = self.game.players[agent_id]

        if not agent.alive:
            return np.zeros(14, dtype=np.int8)

        # Target Mask
        # 죽은 사람은 타겟 불가 (단, 의사는 죽은 사람 살리기 불가, 경찰은 죽은 사람 조사 불가 등 규칙에 따라)
        # 여기서는 간단히 죽은 사람 마스킹
        for i, p in enumerate(status.players):
            if not p.alive:
                mask[i] = 0

        # Role Mask
        # 시민은 Role Claim 외에 Role Action 불가? -> 여기서는 Role Claim 용도로만 Role Head 사용
        # Role Head는 Claim할 직업을 선택하는 것.
        # 0~3: Role, 4: None (Claim 안함)

        return mask
