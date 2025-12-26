ROLE_CITIZEN = 0
ROLE_POLICE = 1
ROLE_DOCTOR = 2
ROLE_MAFIA = 3

PHASE_DAY_DISCUSSION = "day_discussion"
PHASE_DAY_VOTE = "day_vote"
PHASE_NIGHT = "night"

PLAYER_COUNT = 8
MAX_DAYS = 20
ROLES = [
    ROLE_MAFIA,
    ROLE_MAFIA,
    ROLE_POLICE,
    ROLE_DOCTOR,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
    ROLE_CITIZEN,
]

ACTION_SILENT = 0
ACTION_CLAIM_CITIZEN = 1
ACTION_CLAIM_POLICE = 2
ACTION_CLAIM_DOCTOR = 3
ACTION_CLAIM_MAFIA = 4
ACTION_ACCUSE_START = 5
ACTION_ACCUSE_END = ACTION_ACCUSE_START + (PLAYER_COUNT * 4) - 1
TOTAL_ACTIONS = ACTION_ACCUSE_END + 1
LR = 0.00005  # Learning Rate (0.0001 → 0.00005로 하향, 학습 안정성 강화)
GAMMA = 0.99  # Discount Factor (장기 보상 고려)
EPS_CLIP = 0.2  # PPO Clip range (정책 업데이트 제한)
K_EPOCHS = 4  # Update epochs (중간 값 유지)

# === [배치 크기 대폭 증가] ===
# 복잡한 관측 공간(152차원)과 확장된 액션 공간(21개)에 대응
# 더 많은 샘플로 안정적인 gradient 계산
BATCH_SIZE = 256  # 64 → 256으로 대폭 증가 (4배)

# === [탐험-활용 균형 최적화] ===
# Entropy Coefficient 감소: 학습된 정책 활용 강화
# 정책 수렴을 위해 탐험보다 활용에 집중
ENTROPY_COEF = 0.01  # 0.05 → 0.01로 감소 (활용 강화, 정책 수렴 촉진)

# Value Loss 및 Gradient Clipping
VALUE_LOSS_COEF = 0.5  # Value loss coefficient (안정성 유지)
MAX_GRAD_NORM = 0.5  # Gradient clipping (폭발 방지)

# 경로 설정
LOG_DIR = "./logs"
MODEL_DIR = "./models"
