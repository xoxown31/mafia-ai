"""
Engine API 단일화 - 액션 타입 정의

Lunar Lander 스타일: 엔진은 클래스를 모르고 오직 (Type, Target, Value) 신호만 처리
"""
from enum import IntEnum
from typing import Tuple, Optional
from config import Role


class ActionType(IntEnum):
    """액션 유형"""
    NO_ACTION = 0      # 기권
    TARGET_ONLY = 1    # 단순 지목 (토론/투표/밤 행동)
    CLAIM_ROLE = 2     # 역할 주장
    CLAIM_WITH_TARGET = 3  # 역할 주장 + 지목 복합 (경찰 조사 결과 발표)


# Engine이 받는 액션: 단순 튜플 (Type, Target, Value)
EngineAction = Tuple[ActionType, int, Optional[Role]]


class ActionTranslator:
    """
    에이전트별 액션을 Engine 표준 포맷으로 변환하는 Translator 레이어
    
    모든 '해석'은 에이전트 내부에서 끝내고, 엔진에는 정제된 데이터만 전달
    """
    
    @staticmethod
    def to_engine_action(action_data) -> EngineAction:
        """
        다양한 입력 형식을 Engine 표준 포맷으로 통일
        
        Args:
            action_data: 딕셔너리 또는 튜플 형태의 액션 데이터
        
        Returns:
            (ActionType, target_id, role_value) 튜플
        """
        if isinstance(action_data, tuple) and len(action_data) == 3:
            # 이미 Engine 포맷인 경우
            return action_data
        
        if isinstance(action_data, dict):
            return ActionTranslator._dict_to_engine_action(action_data)
        
        # 기본값: 기권
        return (ActionType.NO_ACTION, -1, None)
    
    @staticmethod
    def _dict_to_engine_action(action_dict: dict) -> EngineAction:
        """
        딕셔너리 형태의 액션을 Engine 포맷으로 변환
        
        딕셔너리 구조:
        - target_id: int (지목 대상, -1은 없음)
        - role: Optional[Role] (주장하는 역할)
        - discussion_status: str (토론 종료 여부, "End"면 기권)
        """
        target_id = action_dict.get("target_id", -1)
        role = action_dict.get("role")
        discussion_status = action_dict.get("discussion_status", "Continue")
        
        # 토론 종료 → 기권
        if discussion_status == "End":
            return (ActionType.NO_ACTION, -1, None)
        
        # 역할 주장 + 지목 복합
        if role is not None and target_id != -1:
            return (ActionType.CLAIM_WITH_TARGET, target_id, role)
        
        # 역할 주장만
        if role is not None:
            return (ActionType.CLAIM_ROLE, -1, role)
        
        # 단순 지목
        if target_id != -1:
            return (ActionType.TARGET_ONLY, target_id, None)
        
        # 기권
        return (ActionType.NO_ACTION, -1, None)


class RLActionMapper:
    """
    RL 에이전트용: int(0~20) -> (Type, Target, Value) 매핑 테이블
    
    액션 공간:
    - 0~7: 단순 지목 (TARGET_ONLY)
    - 8: 기권 (NO_ACTION)
    - 9~12: 역할 주장 (CLAIM_ROLE) - 시민/경찰/의사/마피아
    - 13~20: 경찰 주장 + 지목 복합 (CLAIM_WITH_TARGET)
    """
    
    # 정적 매핑 테이블 (고정)
    MAPPING_TABLE = {
        # 0~7: 단순 지목
        0: (ActionType.TARGET_ONLY, 0, None),
        1: (ActionType.TARGET_ONLY, 1, None),
        2: (ActionType.TARGET_ONLY, 2, None),
        3: (ActionType.TARGET_ONLY, 3, None),
        4: (ActionType.TARGET_ONLY, 4, None),
        5: (ActionType.TARGET_ONLY, 5, None),
        6: (ActionType.TARGET_ONLY, 6, None),
        7: (ActionType.TARGET_ONLY, 7, None),
        
        # 8: 기권
        8: (ActionType.NO_ACTION, -1, None),
        
        # 9~12: 역할 주장
        9: (ActionType.CLAIM_ROLE, -1, Role.CITIZEN),
        10: (ActionType.CLAIM_ROLE, -1, Role.POLICE),
        11: (ActionType.CLAIM_ROLE, -1, Role.DOCTOR),
        12: (ActionType.CLAIM_ROLE, -1, Role.MAFIA),
        
        # 13~20: 경찰 주장 + 지목 복합
        13: (ActionType.CLAIM_WITH_TARGET, 0, Role.POLICE),
        14: (ActionType.CLAIM_WITH_TARGET, 1, Role.POLICE),
        15: (ActionType.CLAIM_WITH_TARGET, 2, Role.POLICE),
        16: (ActionType.CLAIM_WITH_TARGET, 3, Role.POLICE),
        17: (ActionType.CLAIM_WITH_TARGET, 4, Role.POLICE),
        18: (ActionType.CLAIM_WITH_TARGET, 5, Role.POLICE),
        19: (ActionType.CLAIM_WITH_TARGET, 6, Role.POLICE),
        20: (ActionType.CLAIM_WITH_TARGET, 7, Role.POLICE),
    }
    
    @staticmethod
    def action_index_to_engine(action_idx: int) -> EngineAction:
        """
        액션 인덱스를 Engine 포맷으로 변환
        
        Args:
            action_idx: 0~20 범위의 액션 인덱스
        
        Returns:
            (ActionType, target_id, role_value) 튜플
        """
        return RLActionMapper.MAPPING_TABLE.get(
            action_idx,
            (ActionType.NO_ACTION, -1, None)  # 범위 외: 기권
        )
