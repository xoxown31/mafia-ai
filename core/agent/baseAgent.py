from abc import *
from typing import List, Dict, Any
import numpy as np
import random
from config import config, Role
from state import GameStatus, GameEvent, GameAction


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

        # 게임 히스토리 추적
        self.vote_history = [0] * config.game.PLAYER_COUNT
        self.char_name = self.__class__.__name__

    @abstractmethod
    def get_action(self, status: GameStatus) -> GameAction:
        """주관적 추론(Hunch) 및 결정 - EngineAction 튜플 반환"""
        pass
