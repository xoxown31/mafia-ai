from abc import *
from typing import List, Dict, Any
import numpy as np
import random
from config import config, Role
from core.engine.state import GameStatus, GameEvent, GameAction


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
