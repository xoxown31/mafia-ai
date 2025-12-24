import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config


class MafiaEnv(gym.Env):
    def __init__(self, log_file=None):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame(log_file=log_file)
        # Action Space: 0~7번 플레이어 지목
        self.action_space = spaces.Discrete(config.PLAYER_COUNT)

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.PLAYER_COUNT + 4,),
                    dtype=np.float32,  # 8 + 4
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(config.PLAYER_COUNT,), dtype=np.int8  # 0 or 1
                ),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        return self._encode_observation(status), {}

    def step(self, action):
        status, done, win = self.game.process_turn(action)

        # 보상 계산
        reward = 0.0
        if status["alive_status"][0]:
            reward += 0.1
        else:
            reward -= 0.1

        if done:
            reward += 10.0 if win else -10.0

        return self._encode_observation(status), reward, done, False, {}

    def _get_action_mask(self):
        mask = np.ones(config.PLAYER_COUNT, dtype=np.int8)  # 초기에는 모두 지목 가능
        my_id = self.game.players[0].id  # 내 플레이어 ID
        my_role = self.game.players[my_id].role  # 내 직업 ID
        phase = self.game.phase  # 현재 게임 단계

        for i in range(config.PLAYER_COUNT):
            # 1. 이미 죽은 플레이어는 지목 불가
            if not self.game.players[i].alive:
                mask[i] = 0
                continue

            # 1. 낮 행동 제약
            if phase == config.PHASE_DAY_VOTE:
                if i == my_id:
                    mask[i] = 0

            # 2. 밤 행동 제약
            if phase == config.PHASE_NIGHT:
                # 밤에는 자기 자신 지목 불가
                if my_role == config.ROLE_MAFIA:
                    if self.game.players[i].role == config.ROLE_MAFIA:
                        mask[i] = 0

                elif my_role == config.ROLE_POLICE:
                    if i == my_id:
                        mask[i] = 0

        return mask

    def _encode_observation(self, status):
        # 1. 관찰 정보 생성
        alive_vector = np.array(status["alive_status"], dtype=np.float32)
        my_role_id = status["roles"][status["id"]]
        role_one_hot = np.zeros(4, dtype=np.float32)
        role_one_hot[my_role_id] = 1.0
        observation = np.concatenate([alive_vector, role_one_hot])

        # 2. 액션 마스크 생성
        action_mask = self._get_action_mask()

        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        # 보기 좋게 출력
        phase_str = (
            ["Claim", "Discussion", "Vote", "Night"][self.game.phase]
            if isinstance(self.game.phase, int)
            else self.game.phase
        )
        alive_indices = [i for i, alive in enumerate(self.game.alive_status) if alive]
        print(f"[Day {self.game.day_count}] {phase_str} | Alive: {alive_indices}")
