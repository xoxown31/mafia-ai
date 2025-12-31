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

class EventType(IntEnum):
    CLAIM = 0
    VOTE = 1
    POLICE_RESULT = 2
    KILL = 3
    EXECUTE = 4
    PROTECT = 5

class GameSettings(BaseSettings):
    PLAYER_COUNT: int = 8
    MAX_DAYS: int = 20
    DEFAULT_ROLES: List[Role] = [
        Role.MAFIA, Role.MAFIA, 
        Role.POLICE, Role.DOCTOR,
        Role.CITIZEN, Role.CITIZEN, Role.CITIZEN, Role.CITIZEN
    ]

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