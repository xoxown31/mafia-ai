from enum import IntEnum
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Role(IntEnum):
    CITIZEN = 0
    POLICE = 1
    DOCTOR = 2
    MAFIA = 3


class Phase(IntEnum):
    DAY_DISCUSSION = 0
    DAY_VOTE = 1
    DAY_EXECUTE = 2
    NIGHT = 3
    GAME_START = 4
    GAME_END = 5


class ActionType(IntEnum):
    """통합 액션 타입 - 에이전트의 의도를 표현"""

    PASS = 0  # 침묵 또는 기권
    TARGET_ACTION = 1  # 지목이 포함된 모든 행동 (투표, 킬, 조사, 치료 등)
    CLAIM = 2  # 정체 주장 (자기 주장 및 타인 지정 주장)


class EventType(IntEnum):
    """게임 이벤트 타입 - 엔진의 기록"""

    SYSTEM_MESSAGE = 0  # 시스템 메시지
    CLAIM = 1  # 주장
    VOTE = 2  # 투표
    POLICE_RESULT = 3  # 경찰 조사 결과
    KILL = 4  # 마피아 킬
    EXECUTE = 5  # 처형
    PROTECT = 6  # 의사 보호


class GameSettings(BaseSettings):
    PLAYER_COUNT: int = 8
    MAX_DAYS: int = 20
    DEFAULT_ROLES: List[Role] = [
        Role.MAFIA,
        Role.MAFIA,
        Role.POLICE,
        Role.DOCTOR,
        Role.CITIZEN,
        Role.CITIZEN,
        Role.CITIZEN,
        Role.CITIZEN,
    ]
    MAX_DISCUSSION_ROUNDS: int = 2
    
    # Model & Env Dimensions
    OBS_DIM: int = 46
    ACTION_DIMS: List[int] = [9, 5]  # [Target(9), Role(5)]


class TrainSettings(BaseSettings):
    LR: float = 0.0001
    GAMMA: float = 0.99
    EPS_CLIP: float = 0.2
    K_EPOCHS: int = 4
    BATCH_SIZE: int = 256
    ENTROPY_COEF: float = 0.05
    VALUE_LOSS_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    IL_COEF: float = 0.1


class PathSettings(BaseSettings):
    LOG_DIR: str = "./logs"
    MODEL_DIR: str = "./models"


class Settings(BaseSettings):
    game: GameSettings = GameSettings()
    train: TrainSettings = TrainSettings()
    paths: PathSettings = PathSettings()

    upstage_api_key: str = Field(..., env="UPSTAGE_API_KEY")

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


config = Settings()
