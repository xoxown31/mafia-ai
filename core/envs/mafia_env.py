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

    def __init__(
        self, 
        render_mode=None, 
        worker_id: Optional[int] = None, 
        log_queue=None,
        id_counter=None,
        id_lock=None
    ):
        self.possible_agents = [f"player_{i}" for i in range(config.game.PLAYER_COUNT)]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode
        
        # [ID Management]
        self._worker_id = worker_id  # If provided explicitly, use it
        self.log_queue = log_queue
        self.id_counter = id_counter
        self.id_lock = id_lock

        # Create dummy agents for the engine
        # MafiaGame expects a list of BaseAgent instances
        self.internal_agents = [EnvAgent(i) for i in range(config.game.PLAYER_COUNT)]
        self.game = MafiaGame(agents=self.internal_agents)

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

        # 상태 추적 변수 (보상 계산용)
        self.last_executed_player = None
        self.last_killed_player = None
        self.last_investigated_player = None
        self.last_protected_player = None
        self.attack_was_blocked = False

    @property
    def worker_id(self):
        """
        Multiprocessing safe lazy-ID loading.
        If initial ID was None, acquire a unique ID from the shared counter.
        This ensures cloned environments in subprocesses get unique IDs.
        """
        if self._worker_id is None:
            if self.id_counter is not None and self.id_lock is not None:
                try:
                    with self.id_lock:
                        self._worker_id = self.id_counter.value
                        self.id_counter.value += 1
                except Exception:
                    # Fallback if lock fails (e.g. not in multiprocessing context properly)
                    self._worker_id = 0
            else:
                self._worker_id = 0
        return self._worker_id

    def _send_log(self, events):
        """Helper to send logs through the multiprocessing queue."""
        if self.log_queue is not None and events:
            try:
                self.log_queue.put((self.worker_id, events))
            except Exception:
                pass

    def reset(self, seed=None, options=None):
        """
        Resets the environment and captures initial game events (Day 0) for external logging.
        """
        self.agents = self.possible_agents[:]
        self.game.reset()

        # Capture initial logs using persistent index
        self.last_history_idx = 0
        new_events = [e.model_dump() for e in self.game.history[self.last_history_idx:]]
        self.last_history_idx = len(self.game.history)

        # 상태 초기화
        self.last_executed_player = None
        self.last_killed_player = None
        self.last_investigated_player = None
        self.last_protected_player = None
        self.attack_was_blocked = False

        observations = {}
        for agent in self.agents:
            pid = self._agent_to_id(agent)
            observations[agent] = {
                "observation": self._encode_observation(pid),
                "action_mask": self._get_action_mask(pid),  # 액션 마스크 포함 필수
            }

        infos = {agent: {} for agent in self.agents}
        
        # [MODIFIED] Send logs to queue instead of info
        if new_events:
            self._send_log(new_events)

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
        prev_phase = self.game.phase
        prev_alive_count = sum(1 for p in self.game.players if p.alive)

        # 게임 진행
        status, is_over, is_win = self.game.step_phase(engine_actions)

        # Capture new logs
        new_events = [e.model_dump() for e in self.game.history[self.last_history_idx:]]
        self.last_history_idx = len(self.game.history)

        # 상태 변화 추적
        self._track_state_changes(prev_phase, prev_alive_count, engine_actions)

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

            rewards[agent] = self._calculate_reward(
                pid, prev_phase, engine_actions.get(pid), is_over, my_win
            )
            terminations[agent] = is_over
            truncations[agent] = False
            
            agent_info = {"day": status.day, "phase": status.phase, "win": my_win}
            infos[agent] = agent_info
        
        # [MODIFIED] Send logs to queue
        if new_events:
            self._send_log(new_events)

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

    def get_game_status(self, agent_id):
        """
        특정 에이전트 시점의 게임 상태를 반환합니다.
        runner.py 등 외부에서 에이전트의 행동 결정을 위해 호출합니다.
        """
        return self.game.get_game_status(agent_id)

    def _agent_to_id(self, agent_str):
        return int(agent_str.split("_")[1])

    def _id_to_agent(self, agent_id):
        return f"player_{agent_id}"

    # === 상태 추적 ===

    def _track_state_changes(self, prev_phase, prev_alive_count, actions):
        """
        이전 페이즈와 현재 상태를 비교하여 중요한 이벤트를 추적
        """
        current_alive_count = sum(1 for p in self.game.players if p.alive)

        # 처형 추적
        if prev_phase == Phase.DAY_VOTE and self.game.phase == Phase.DAY_EXECUTE:
            # 투표로 처형된 플레이어 찾기
            for player in self.game.players:
                if not player.alive and player.vote_count > 0:
                    self.last_executed_player = player
                    break

        # 밤 사망 추적
        if prev_phase == Phase.NIGHT and current_alive_count < prev_alive_count:
            # 마피아에게 죽은 플레이어 찾기
            for player in self.game.players:
                if not player.alive and player != self.last_executed_player:
                    self.last_killed_player = player
                    break

        # 보호 성공 추적 (밤인데 아무도 안 죽음)
        if prev_phase == Phase.NIGHT and current_alive_count == prev_alive_count:
            mafia_target = None
            doctor_target = None

            for pid, action in actions.items():
                player = self.game.players[pid]
                if not player.alive:
                    continue

                target_id = -1
                if isinstance(action, dict):
                    target_id = action.get("target_id", -1)
                elif hasattr(action, "target_id"):
                    target_id = action.target_id

                if target_id != -1 and 0 <= target_id < len(self.game.players):
                    if player.role == Role.MAFIA:
                        mafia_target = target_id
                    elif player.role == Role.DOCTOR:
                        doctor_target = target_id

            # 마피아 타겟과 의사 타겟이 같으면 보호 성공
            if mafia_target is not None and doctor_target is not None:
                if mafia_target == doctor_target:
                    self.attack_was_blocked = True
                    self.last_protected_player = self.game.players[doctor_target]
        else:
            self.attack_was_blocked = False
            self.last_protected_player = None

    # === 단순화된 보상 시스템 ===

    def _calculate_reward(self, agent_id, prev_phase, my_action, done, win):
        """
        3가지 핵심 보상만 사용:
        1. 게임 승패 (주요 신호)
        2. 마피아 제거 기여
        3. 역할별 핵심 목표 달성
        """
        reward = 0.0
        agent = self.game.players[agent_id]

        # === 보상 1: 게임 승패 (가장 중요) ===
        if done:
            reward += 10.0 if win else -10.0

        # === 보상 2: 마피아 제거 기여 ===
        if prev_phase == Phase.DAY_VOTE and self.last_executed_player:
            reward += self._calculate_execution_reward(agent_id, my_action)

        # === 보상 3: 역할별 핵심 목표 달성 ===
        if prev_phase == Phase.NIGHT:
            reward += self._calculate_night_action_reward(agent_id, my_action)

        return reward

    def _calculate_execution_reward(self, agent_id, my_action):
        """
        처형 단계에서 보상 계산
        - 마피아 처형 성공 시 투표한 시민팀에게 보상
        - 시민 처형 시 페널티
        """
        if not self.last_executed_player:
            return 0.0

        reward = 0.0
        agent = self.game.players[agent_id]
        executed = self.last_executed_player

        # 내가 이 플레이어에게 투표했는가?
        did_vote = False
        if my_action:
            target_id = -1
            if isinstance(my_action, dict):
                target_id = my_action.get("target_id", -1)
            elif hasattr(my_action, "target_id"):
                target_id = my_action.target_id

            did_vote = target_id == executed.id

        if not did_vote:
            return 0.0

        # 역할에 따른 보상
        if agent.role != Role.MAFIA:
            # 시민팀: 마피아 처형 시 보상
            if executed.role == Role.MAFIA:
                reward += 4.0
            # 중요 역할 처형 시 큰 페널티
            elif executed.role in [Role.POLICE, Role.DOCTOR]:
                reward -= 3.0
            # 일반 시민 처형 시 작은 페널티
            elif executed.role == Role.CITIZEN:
                reward -= 1.0
        else:
            # 마피아: 시민 처형 시 보상, 동료 마피아 처형 시 페널티
            if executed.role == Role.MAFIA:
                reward -= 5.0  # 동료 배신
            elif executed.role in [Role.POLICE, Role.DOCTOR]:
                reward += 2.0  # 중요 역할 제거
            else:
                reward += 1.0  # 일반 시민 제거

        return reward

    def _calculate_night_action_reward(self, agent_id, my_action):
        """
        밤 행동 보상 계산
        - 마피아: 성공적인 킬
        - 경찰: 마피아 발견
        - 의사: 보호 성공
        """
        if not my_action:
            return 0.0

        reward = 0.0
        agent = self.game.players[agent_id]

        target_id = -1
        if isinstance(my_action, dict):
            target_id = my_action.get("target_id", -1)
        elif hasattr(my_action, "target_id"):
            target_id = my_action.target_id

        if target_id == -1 or target_id >= len(self.game.players):
            return 0.0

        target = self.game.players[target_id]

        # 마피아: 킬 성공
        if agent.role == Role.MAFIA:
            if self.last_killed_player and self.last_killed_player.id == target_id:
                # 중요 역할 제거 시 더 큰 보상
                if target.role == Role.POLICE:
                    reward += 4.0
                elif target.role == Role.DOCTOR:
                    reward += 3.0
                else:
                    reward += 2.0

        # 경찰: 마피아 조사 성공
        elif agent.role == Role.POLICE:
            if target.role == Role.MAFIA and target.alive:
                reward += 3.0

        # 의사: 보호 성공
        elif agent.role == Role.DOCTOR:
            if self.attack_was_blocked and self.last_protected_player:
                if self.last_protected_player.id == target_id:
                    # 중요 역할 보호 시 더 큰 보상
                    if target.role == Role.POLICE:
                        reward += 4.0
                    elif target.role == Role.DOCTOR:
                        reward += 3.0
                    else:
                        reward += 2.5

        return reward

    # === 관측 및 액션 마스크 ===

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
        obs = np.concatenate(
            [
                id_vec,  # 8
                role_vec,  # 4
                day_vec,  # 1
                phase_vec,  # 3
                actor_vec,  # 9
                target_vec,  # 9
                value_vec,  # 5
                type_vec,  # 7
            ]
        )  # Total 46

        return obs

    def _get_action_mask(self, agent_id):
        """
        에이전트가 현재 상태에서 수행할 수 있는 유효한 행동 마스크를 생성합니다.
        마스크는 [Target(9), Role(5)] 형태로 총 14차원입니다.

        타겟 마스크 (_target_mask, 9차원):
            - mask[0]: 아무도 지목하지 않음 (PASS)
            - mask[1] ~ mask[8]: Player 0 ~ 7 지목

        역할 마스크 (_role_mask, 5차원):
            - mask[0]: 역할을 주장하지 않음
            - mask[1] ~ mask[4]: 시민, 경찰, 의사, 마피아 역할 주장
        """
        status = self.game.get_game_status(agent_id)
        agent = self.game.players[agent_id]

        _target_mask = np.zeros(9, dtype=np.int8)
        _role_mask = np.zeros(5, dtype=np.int8)

        if not agent.alive:
            return np.concatenate([_target_mask, _role_mask])

        _target_mask[0] = 1
        phase = status.phase
        is_active_night_role = phase == Phase.NIGHT and agent.role in [
            Role.MAFIA,
            Role.POLICE,
            Role.DOCTOR,
        ]

        if is_active_night_role:
            _target_mask[0] = 0
        else:
            _target_mask[0] = 1

        _role_mask[0] = 1

        valid_targets = {p.id for p in self.game.players if p.alive}
        phase = status.phase

        if phase == Phase.NIGHT:
            if agent.role == Role.MAFIA:
                mafia_team_ids = {
                    p.id for p in self.game.players if p.role == Role.MAFIA
                }
                valid_targets -= mafia_team_ids
            elif agent.role == Role.POLICE:
                valid_targets.discard(agent_id)
            elif agent.role == Role.CITIZEN:
                valid_targets.clear()

        elif phase == Phase.DAY_VOTE:
            valid_targets.discard(agent_id)

        elif phase == Phase.DAY_DISCUSSION:
            _role_mask[1:] = 1

        if not valid_targets:
            _target_mask[1:] = 0
        else:
            for target_id in valid_targets:
                _target_mask[target_id + 1] = 1

        return np.concatenate([_target_mask, _role_mask])
