import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config

class MafiaEnv(gym.Env):
    """
    MafiaGame을 강화학습 환경으로 래핑하는 클래스
    """
    def __init__(self):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame()
        
        # Action Space: 0~7번 플레이어 지목
        self.action_space = spaces.Discrete(config.PLAYER_COUNT)
        
        # Observation Space: (예시) 상태 벡터 크기 128
        # 실제 구현 시 _encode_observation 결과 크기에 맞춰 수정 필요
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        return self._encode_observation(status), {}

    def step(self, action):
        status, done, win = self.game.process_turn(action)
        
        # 보상 계산 (Reward Shaping)
        reward = 1.0 if win else (-1.0 if done else 0.0)
        
        truncated = False
        return self._encode_observation(status), reward, done, truncated, {}

    def _encode_observation(self, status: dict) -> np.ndarray:
        # TODO: 딕셔너리 상태를 신경망 입력용 벡터(np.array)로 변환
        return np.zeros(128, dtype=np.float32)

    def render(self):
        print(f"[Day {self.game.day_count}] {self.game.phase}")