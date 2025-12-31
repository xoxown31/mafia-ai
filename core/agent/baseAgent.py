from abc import *
from typing import List
import numpy as np
import random
from config import config, Role
from state import GameStatus, GameEvent


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
    def __init__(self, player_id: int, role: Role = Role.CITIZEN):
        self.id = player_id
        self.role = role
        self.alive = True
        self.current_status: GameStatus = None

        # Belief Matrix: (N x 4) - 각 플레이어가 각 직업일 것이라는 신뢰 점수
        # 열(Col): [0: 시민, 1: 경찰, 2: 의사, 3: 마피아]
        self.belief = np.random.normal(0, 0.1, (config.game.PLAYER_COUNT, len(Role)))

        # 자신의 belief는 확실하게 설정
        role_idx = int(self.role)
        self.belief[self.id, :] = -100.0
        self.belief[self.id, role_idx] = 100.0

        # 게임 히스토리 추적
        self.vote_history = [0] * config.game.PLAYER_COUNT
        self.char_name = self.__class__.__name__

    def observe(self, status: GameStatus):
        """정보를 수신하는 유일한 입구"""
        self.current_status = status
        self.update_belief(status.action_history)

    @abstractmethod
    def update_belief(self, history: List[GameEvent]):
        """객관적 사실(Fact) 반영"""
        pass

    @abstractmethod
    def get_action(self, conversation_log: str) -> str:
        """주관적 추론(Hunch) 및 결정"""
        pass
