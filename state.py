from pydantic import BaseModel
from typing import List, Optional, Union
from config import Role, Phase, EventType, ActionType


class PlayerStatus(BaseModel):
    id: int
    alive: bool


class GameAction(BaseModel):
    """
    에이전트의 의도를 담은 액션 모델 (Agent → Engine)

    Multi-Discrete 구조: [Type, Target, Role]
    - action_type: PASS(0), TARGET_ACTION(1), CLAIM(2)
    - target_id: 지목 대상 플레이어 ID (-1은 없음, 0~7)
    - claim_role: 주장하는 역할 (None 또는 Role enum)
    """

    action_type: ActionType
    target_id: int = -1  # 기본값: 지목 없음
    claim_role: Optional[Role] = None  # 기본값: 주장 없음

    def to_multi_discrete(self) -> List[int]:
        """Multi-Discrete 벡터로 변환: [type, target, role]"""
        role_value = int(self.claim_role) if self.claim_role is not None else -1
        return [int(self.action_type), self.target_id, role_value]

    @classmethod
    def from_multi_discrete(cls, vector: List[int]) -> "GameAction":
        """Multi-Discrete 벡터에서 생성: [type, target, role]"""
        action_type = ActionType(vector[0])
        target_id = vector[1]
        role_value = vector[2]
        claim_role = Role(role_value) if role_value >= 0 else None
        return cls(action_type=action_type, target_id=target_id, claim_role=claim_role)


class GameEvent(BaseModel):
    """
    게임 엔진의 기록 (Engine → History)

    EventType 규격을 공유하되, MafiaAction과 분리하여 관리
    """

    day: int
    phase: Phase
    event_type: EventType
    actor_id: int
    target_id: Optional[int] = None
    value: Union[Role, bool, None] = None


class GameStatus(BaseModel):
    day: int
    phase: Phase
    my_id: int
    my_role: Role
    players: List[PlayerStatus]
    action_history: List[GameEvent]
