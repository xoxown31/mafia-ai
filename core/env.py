import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config


status = {
    "alive_status": [True, True, False, True, True, False, True, True],
    "roles": [0, 0, 0, 0, 1, 2, 3, 3],
    "day_count": 3,
    "id": 0,
}


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
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(128,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        return self._encode_observation(status), {}

    def step(self, action):
        status, done, win = self.game.process_turn(action)

        reward = 0.0
        if status["alive_status"][status["id"]] is False:
            reward -= 0.1  # 죽으면 패널티
        else:
            reward += 0.1  # 살아있으면 소소한 보상

        # 보상 계산 (Reward Shaping)
        reward += 1.0 if win else (-1.0 if done else 0.0)

        truncated = False
        return self._encode_observation(status), reward, done, truncated, {}

    def _encode_observation(self, status: dict) -> np.ndarray:
        # TODO: 딕셔너리 상태를 신경망 입력용 벡터(np.array)로 변환
        alive_vector = np.array(
            status["alive_status"], dtype=np.float32
        )  # 살아있는 플레이어 상태 벡터

        my_role_id = status["roles"][status["id"]]  # 내 역할 ID
        role_one_hot = np.zeros(4, dtype=np.float32)  # 역할 수에 맞게 원-핫 인코딩
        role_one_hot[my_role_id] = 1.0  # 내 역할 원-핫 인코딩

        observation = np.concatenate([alive_vector, role_one_hot])
        return observation

    def render(self):
        print(f"[Day {self.game.day_count}] {self.game.phase}")


if __name__ == "__main__":
    env = MafiaEnv()
    obs = env._encode_observation(status)
    print("Encoded Observation: ", obs)
