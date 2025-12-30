from pydantic import BaseModel
from typing import List, Optional, Any
from config import Role, Phase, EventType

class PlayerStatus(BaseModel):
    id: int
    is_alive: bool

class GameEvent(BaseModel):
    day: int
    phase: Phase
    event_type: EventType
    actor_id: int
    target_id: Optional[int] = None
    value: Any = None

class GameStatus(BaseModel):
    day: int
    phase: Phase
    my_id: int
    my_role: Role
    players: List[PlayerStatus]
    action_history: List[GameEvent]