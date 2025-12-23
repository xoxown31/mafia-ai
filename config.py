# config.py

# 역할 지정
ROLE_CITIZEN = 0
ROLE_POLICE = 1
ROLE_DOCTOR = 2
ROLE_MAFIA = 3

# 게임 설정
PLAYER_COUNT = 8
ROLES = [
    "Mafia",
    "Mafia",
    "Police",
    "Doctor",
    "Citizen",
    "Citizen",
    "Citizen",
    "Citizen",
]

# 학습 설정 (Hyperparameters)
LR = 0.0003  # Learning Rate
GAMMA = 0.99  # Discount Factor
EPS_CLIP = 0.2  # PPO Clip range
K_EPOCHS = 4  # Update epochs
BATCH_SIZE = 32

# 경로 설정
LOG_DIR = "./logs"
MODEL_DIR = "./models"
