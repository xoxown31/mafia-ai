from abc import *
from typing import List
import numpy as np
import random
import config


# Softmax 유틸리티 함수
def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax 함수: 점수(logits)를 확률 분포로 변환
    temperature: 낮을수록 결정론적, 높을수록 균등 분포
    """
    # 오버플로우 방지
    scores = scores / temperature
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    return exp_scores / np.sum(exp_scores)


class BaseAgent(ABC):
    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        self.id = player_id
        self.role = role
        self.alive = True
        self.claimed_target = -1
        self.claimed_role = -1  # 주장한 역할 (-1: 주장 없음, 0~3: 역할)

        # Belief Matrix: (N x 4) - 각 플레이어가 각 직업일 것이라는 신뢰 점수
        # 열(Col): [0: 시민, 1: 경찰, 2: 의사, 3: 마피아]
        self.belief = np.zeros((config.PLAYER_COUNT, 4), dtype=np.float32)

        # 자신의 belief는 확실하게 설정
        self.belief[self.id, self.role] = 100.0

        # 게임 히스토리 추적
        self.voted_by_last_turn = []
        self.vote_history = [0] * config.PLAYER_COUNT
        self.investigated_players = set()  # 경찰이 조사한 플레이어 추적
        self.confirmed_mafia = set()  # 경찰이 확인한 마피아
        self.should_reveal = False  # 경찰이 공개할지 여부
        self.char_name = self.__class__.__name__

    @abstractmethod
    def update_belief(self, game_status: dict):
        pass

    @abstractmethod
    def get_action(self, players: List["BaseAgent"], current_role: int):
        pass
