import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config

class MafiaEnv(gym.Env):
    def __init__(self):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame()
        
        # Action Space: 0~7번 플레이어 지목
        self.action_space = spaces.Discrete(config.PLAYER_COUNT)
        
        # [수정] Observation Space 크기 계산
        # 생존자(8) + 내 직업 원핫(4) = 12
        obs_dim = config.PLAYER_COUNT + 4 
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        return self._encode_observation(status), {}

    def step(self, action):
        status, done, win = self.game.process_turn(action)
        
        # 보상 계산 (Reward Shaping)
        reward = 0.0
        # 1. 생존 보상 (작게)
        if status["alive_status"][0]: # 0번(AI)이 살아있으면
            reward += 0.1
        else:
            reward -= 0.1

        # 2. 승패 보상 (크게)
        if done:
            reward += (10.0 if win else -10.0) # 승리 보상을 좀 더 크게 줌
        
        truncated = False
        return self._encode_observation(status), reward, done, truncated, {}

    def _encode_observation(self, status: dict) -> np.ndarray:
        # 1. 생존자 정보 (8개)
        alive_vector = np.array(status["alive_status"], dtype=np.float32)
        
        # 2. 내 직업 정보 One-Hot Encoding (4개)
        my_role_id = status["roles"][status["id"]]
        role_one_hot = np.zeros(4, dtype=np.float32)
        role_one_hot[my_role_id] = 1.0
        
        # 이어 붙이기 (Concatenate) -> 총 12개
        observation = np.concatenate([alive_vector, role_one_hot])
        return observation

    def render(self):
        # 보기 좋게 출력
        phase_str = ["Claim", "Discussion", "Vote", "Night"][self.game.phase] if isinstance(self.game.phase, int) else self.game.phase
        alive_indices = [i for i, alive in enumerate(self.game.alive_status) if alive]
        print(f"[Day {self.game.day_count}] {phase_str} | Alive: {alive_indices}")