"""
관측(Observation) 공간의 탈(脫) 레거시

Lunar Lander 스타일: 복잡한 클래스 대신 history(List[GameEvent])만 사용하여
상태 행렬(State Matrix)을 만드는 순수 함수
"""
import numpy as np
from typing import List
from config import config, Role, Phase, EventType
from state import GameEvent, GameStatus


def compute_state_matrix(
    status: GameStatus,
    viewer_id: int,
    last_vote_record: np.ndarray
) -> np.ndarray:
    """
    GameStatus와 history만으로 152차원 상태 벡터를 생성하는 순수 함수
    
    레거시 로직(claimed_list, accusation_matrix 등)을 버리고
    오직 history(List[GameEvent])만 훑어서 상태를 계산
    
    State Matrix 구성 (총 152차원):
    - alive_status: 8 (생존 여부)
    - my_role: 4 (내 역할 one-hot)
    - claim_status: 8 (각 플레이어가 주장한 역할: 0~3)
    - accusation_matrix: 8*8=64 (누가 누구를 의심했는지)
    - last_vote_matrix: 8*8=64 (직전 투표에서 누가 누구에게 투표했는지)
    - day_count: 1 (현재 날짜, 정규화)
    - phase_onehot: 3 (현재 페이즈: discussion, vote, night)
    
    Args:
        status: 게임 상태
        viewer_id: 관찰자 ID
        last_vote_record: 이전 투표 기록 (8x8 행렬)
    
    Returns:
        152차원 numpy 배열
    """
    n_players = config.game.PLAYER_COUNT
    
    # 1. alive_status (8차원)
    alive_status = np.array([
        1.0 if p.alive else 0.0 
        for p in status.players
    ], dtype=np.float32)
    
    # 2. my_role (4차원 one-hot)
    my_role = np.zeros(len(Role), dtype=np.float32)
    my_role[int(status.my_role)] = 1.0
    
    # 3. claim_status (8차원) - history에서 CLAIM 이벤트 추출
    claim_status = np.zeros(n_players, dtype=np.float32)
    for event in status.action_history:
        if event.event_type == EventType.CLAIM and event.value is not None:
            if isinstance(event.value, Role):
                claim_status[event.actor_id] = float(int(event.value))
    
    # 4. accusation_matrix (64차원 = 8x8) - CLAIM + target_id가 있는 경우
    accusation_matrix = np.zeros((n_players, n_players), dtype=np.float32)
    for event in status.action_history:
        if event.event_type == EventType.CLAIM and event.target_id is not None:
            accusation_matrix[event.actor_id][event.target_id] = 1.0
    accusation_flat = accusation_matrix.flatten()
    
    # 5. last_vote_matrix (64차원 = 8x8) - 외부에서 전달받은 투표 기록
    last_vote_flat = last_vote_record.flatten()
    
    # 6. day_count (1차원, 정규화)
    day_normalized = np.array([status.day / config.game.MAX_DAYS], dtype=np.float32)
    
    # 7. phase_onehot (3차원)
    phase_onehot = np.zeros(3, dtype=np.float32)
    if status.phase == Phase.DAY_DISCUSSION:
        phase_onehot[0] = 1.0
    elif status.phase == Phase.DAY_VOTE:
        phase_onehot[1] = 1.0
    elif status.phase == Phase.NIGHT:
        phase_onehot[2] = 1.0
    
    # 전체 결합
    state_vector = np.concatenate([
        alive_status,       # 8
        my_role,            # 4
        claim_status,       # 8
        accusation_flat,    # 64
        last_vote_flat,     # 64
        day_normalized,     # 1
        phase_onehot,       # 3
    ])
    
    assert state_vector.shape == (152,), f"Expected 152 dims, got {state_vector.shape}"
    return state_vector


def compute_action_mask(
    status: GameStatus,
    my_id: int,
    my_role: Role
) -> np.ndarray:
    """
    현재 상태에서 유효한 액션 마스크를 계산하는 순수 함수
    
    Args:
        status: 게임 상태
        my_id: 관찰자 ID
        my_role: 관찰자 역할
    
    Returns:
        21차원 액션 마스크 (0: 불가능, 1: 가능)
    """
    mask = np.ones(21, dtype=np.int8)
    phase = status.phase
    
    # === 0~7: 지목 액션 마스크 ===
    for i in range(config.game.PLAYER_COUNT):
        # 1. 이미 죽은 플레이어는 지목 불가
        if not status.players[i].alive:
            mask[i] = 0
            continue

        # 2. 낮 행동 제약 (자신 지목 불가)
        if phase in (Phase.DAY_DISCUSSION, Phase.DAY_VOTE):
            if i == my_id:
                mask[i] = 0

        # 3. 밤 행동 제약
        elif phase == Phase.NIGHT:
            # 마피아: 동료 마피아 지목 불가 (여기서는 역할 정보가 없으므로 생략)
            # 경찰: 자신 조사 불가
            if my_role == Role.POLICE:
                if i == my_id:
                    mask[i] = 0

    # === 8: 기권 액션 마스크 ===
    if phase in (Phase.DAY_DISCUSSION, Phase.DAY_VOTE):
        mask[8] = 1  # 기권 허용
    else:
        mask[8] = 0  # 밤에는 기권 불가
    
    # === 9~12: 역할 주장 액션 마스크 (토론 단계에만 가능) ===
    if phase == Phase.DAY_DISCUSSION:
        mask[9:13] = 1  # 시민/경찰/의사/마피아 주장 모두 허용
    else:
        mask[9:13] = 0  # 토론 단계 외에는 주장 불가
    
    # === 13~20: 경찰 주장+지목 복합 액션 (토론 단계에만) ===
    if phase == Phase.DAY_DISCUSSION:
        for i in range(8):
            if status.players[i].alive and i != my_id:
                mask[13 + i] = 1  # 살아있고 자신이 아닌 플레이어 지목 가능
            else:
                mask[13 + i] = 0
    else:
        mask[13:21] = 0  # 토론 단계 외에는 불가

    return mask
