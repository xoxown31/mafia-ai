from pydantic import BaseModel
from typing import List, Optional, Union
from config import Role, Phase, EventType


class PlayerStatus(BaseModel):
    id: int
    alive: bool


class GameAction(BaseModel):
    """
    에이전트의 의도를 담은 액션 모델 (Agent → Engine)
    
    Multi-Discrete 구조: [Target, Role]
    - target_id: 지목 대상 플레이어 ID (-1은 없음, 0~7)
    - claim_role: 주장하는 역할 (None 또는 Role enum)
    
    Action Type은 Target과 Role의 조합으로 추론:
    - PASS: Target=-1, Role=None
    - TARGET_ACTION: Target!=-1, Role=None
    - CLAIM: Role!=None
    """
    target_id: int = -1  # 기본값: 지목 없음
    claim_role: Optional[Role] = None  # 기본값: 주장 없음

    def to_multi_discrete(self) -> List[int]:
        """Multi-Discrete 벡터로 변환: [target, role]"""
        # Target: -1 -> 0, 0~7 -> 1~8
        target_val = self.target_id + 1
        
        # Role: None -> 0, 0~3 -> 1~4
        role_val = int(self.claim_role) + 1 if self.claim_role is not None else 0
        
        return [target_val, role_val]
    
    @classmethod
    def from_multi_discrete(cls, vector: List[int]) -> 'GameAction':
        """Multi-Discrete 벡터에서 생성: [target, role]"""
        target_val = vector[0]
        role_val = vector[1]
        
        target_id = target_val - 1
        claim_role = Role(role_val - 1) if role_val > 0 else None
        
        return cls(target_id=target_id, claim_role=claim_role)


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
