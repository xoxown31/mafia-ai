from abc import ABC, abstractmethod
from typing import List
import math
import config

# 유틸리티 함수: 점수 -> 확률 변환
def sigmoid(x):
    # 오버플로우 방지
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

class BaseCharacter(ABC):
    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        self.id = player_id
        self.role = role
        self.alive = True
        self.claimed_target = -1

        # [변경] suspicion: 확률이 아닌 '점수(Score)'로 관리
        # 0.0 = 중립(확률 0.5), 음수 = 신뢰, 양수 = 의심
        self.suspicion = [0.0 for _ in range(config.PLAYER_COUNT)]

        self.voted_by_last_turn = []
        self.vote_history = [0] * config.PLAYER_COUNT
        self.char_name = self.__class__.__name__

    # [추가] 특정 플레이어에 대한 의심 '확률'을 반환하는 헬퍼
    def get_suspicion_prob(self, player_id: int) -> float:
        return sigmoid(self.suspicion[player_id])

    def _get_alive_ids(
        self, players: List["BaseCharacter"], exclude_me: bool = True
    ) -> List[int]:
        exclude_id = self.id if exclude_me else -1
        return [p.id for p in players if p.alive and p.id != exclude_id]

    @abstractmethod
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        pass

    @abstractmethod
    def update_suspicion(self, speaker: "BaseCharacter", target_id: int):
        pass

    @abstractmethod
    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        pass

    @abstractmethod
    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        pass