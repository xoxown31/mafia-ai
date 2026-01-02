# mafia-ai

강화학습 기반 마피아 게임 AI 에이전트

## 📋 프로젝트 개요

마피아 게임을 강화학습(Reinforcement Learning)으로 플레이하는 AI 에이전트를 개발합니다. 에이전트는 게임 내 공개 정보(발언, 투표)를 관찰하여 역할에 맞는 전략을 학습합니다.

### 주요 특징
- 🎭 **4가지 역할**: 시민, 경찰, 의사, 마피아
- 🧠 **공정 정보 학습**: 게임 내 공개된 정보만 사용
- 🎯 **역할별 전략**: 각 역할에 최적화된 행동 패턴
- 📊 **152차원 관측 공간**: 발언, 투표, 시간 정보 포함

## 🚀 최신 업데이트 (2025-12-25)

### v2.0 - 공정 정보 기반 학습 시스템
- ✅ **관측 공간 확장**: 12차원 → 78차원
  - 발언 정보 (Claim Status)
  - 지목 관계 (Accusation Matrix)
  - 투표 기록 (Vote History)
  - 게임 상태 (Day, Phase)
  
- ✅ **RNN(LSTM/GRU) 기반 시퀀스 모델**: 시계열 데이터 처리 최적화
- ✅ **보상 체계 개선**: 승패 중심 (승리 +100, 패배 -50)
- ✅ **학습 안정화**: Gradient Clipping, Dropout 적용

👉 자세한 내용은 [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md) 및 [COMPARISON.md](COMPARISON.md) 참조

## 👥 Contributors

Name | Role | GitHub
---- | ---- | ------
Taeju Park | Project Manager / System Integration | @xoxown31
Changkwon Kim | Game Engine / Environment Design | @chang1025
Jinwon Hong | AI Researcher / RL Implementation | @11Won11

## 🛠️ 설치 및 실행

### 요구사항
```bash
Python >= 3.8
torch >= 1.9.0
gymnasium >= 0.26.0
numpy >= 1.21.0
```

### 설치
```bash
pip install -r requirements.txt
```

### 학습
```bash
# PPO 에이전트 학습 (권장)
python main.py --mode train --agent ppo --episodes 1000

# REINFORCE 에이전트 학습
python main.py --mode train --agent reinforce --episodes 500
```

### 테스트
```bash
python main.py --mode test --agent ppo
```

## 📊 관측 공간 구조

```python
Observation Space (78차원):
├── 생존 상태 (8)          # 각 플레이어 생존 여부
├── 내 역할 (4)            # One-hot encoding
├── 주장한 역할 (8)        # 플레이어별 역할 주장
├── 지목 관계 (64)         # 8×8 의심 매트릭스
├── 투표 기록 (64)         # 8×8 직전 투표 기록
├── 현재 날짜 (1)          # 정규화된 날짜
└── 페이즈 (3)             # 토론/투표/밤
```

## 🎮 게임 규칙

### 역할
- **시민 (4명)**: 마피아를 찾아 투표로 제거
- **경찰 (1명)**: 밤에 한 명을 조사하여 마피아 여부 확인
- **의사 (1명)**: 밤에 한 명을 치료하여 마피아 공격 방어
- **마피아 (2명)**: 밤에 시민 팀 제거, 낮에는 신분 숨김

### 승리 조건
- **시민 팀**: 모든 마피아 제거
- **마피아 팀**: 마피아 수 ≥ 시민 수

## 📈 보상 체계

| 상황 | 보상 |
|------|------|
| 승리 | +100 |
| 패배 | -50 |
| 조기 승리 | +(MAX_DAYS-day)×2 |
| 마피아 투표 성공 (시민) | +5.0 |
| 경찰 제거 (마피아) | +8~10 |
| 마피아 발견 (경찰) | +8.0 |
| 치료 성공 (의사) | +10~15 |
| 동료 지목 (마피아) | -10.0 |

## 🏗️ 프로젝트 구조

```
mafia-ai/
├── ai/
│   ├── model.py          # Actor-Critic 네트워크 (Deep MLP)
│   ├── ppo.py            # PPO 알고리즘
│   ├── reinforce.py      # REINFORCE 알고리즘
│   └── buffer.py         # Experience 버퍼
├── core/
│   ├── env.py            # Gymnasium 환경 (152차원 관측)
│   ├── game.py           # 마피아 게임 로직
│   ├── runner.py         # 학습/테스트 루프
│   └── characters/
│       ├── base.py       # 캐릭터 기본 클래스
│       └── rational.py   # 휴리스틱 봇
├── utils/
│   ├── analysis.py       # 로그 분석
│   └── visualize.py      # 시각화
├── config.py             # 하이퍼파라미터 설정
├── main.py               # 메인 실행 파일
└── README.md
```

## 🔬 실험 결과

### 학습 지표
- **승률**: 역할별 승률 추이
- **평균 보상**: 에피소드당 누적 보상
- **게임 길이**: 평균 턴 수
- **행동 분포**: 각 행동 선택 빈도

### 분석 도구
```bash
# 로그 분석 및 시각화
python -m utils.analysis logs/mafia_game_log.txt
```

## 🔧 하이퍼파라미터

```python
# config.py
BATCH_SIZE = 64           # 배치 크기
LR = 0.0001               # 학습률
GAMMA = 0.99              # 할인율
EPS_CLIP = 0.2            # PPO Clipping
K_EPOCHS = 4              # 업데이트 에포크
ENTROPY_COEF = 0.01       # 엔트로피 계수
VALUE_LOSS_COEF = 0.5     # Value Loss 가중치
MAX_GRAD_NORM = 0.5       # Gradient Clipping
```

## 📚 참고 자료

- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [마피아 게임 위키](https://en.wikipedia.org/wiki/Mafia_(party_game))

## 🐛 문제 해결

### Q: 학습이 수렴하지 않아요
A: BATCH_SIZE를 64→128로 증가하거나 LR을 낮춰보세요 (0.0001 → 0.00005)

### Q: 메모리 부족 에러
A: BATCH_SIZE를 32로 줄이거나 모델 크기를 축소하세요

### Q: 특정 역할만 학습이 안 돼요
A: 해당 역할의 보상 가중치를 config.py에서 조정하세요

## 📜 라이선스

MIT License

## 🙏 감사의 말

이 프로젝트는 강화학습과 게임 이론을 결합한 연구 프로젝트입니다.