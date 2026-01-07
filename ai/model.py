import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicActorCritic(nn.Module):
    """
    RNN(LSTM/GRU) 기반의 Actor-Critic 모델 (차원 유연성 강화 & 배치 충돌 방지)
    """

    def __init__(
        self,
        state_dim,
        action_dims=[9, 5],
        backbone="lstm",
        hidden_dim=128,
        num_layers=2,
    ):
        super(DynamicActorCritic, self).__init__()

        self.backbone_type = backbone.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.action_dims = action_dims

        # [설정] batch_first=True -> (Batch, Seq, Feature)
        if self.backbone_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
        elif self.backbone_type == "gru":
            self.backbone = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Choose 'lstm' or 'gru'"
            )

        feature_dim = hidden_dim

        # Multi-Discrete Action Heads
        self.actor_target = nn.Linear(feature_dim, action_dims[0])
        self.actor_role = nn.Linear(feature_dim, action_dims[1])
        self.critic = nn.Linear(feature_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, hidden_state=None):
        """
        Args:
            state: 입력 상태
            hidden_state: RNN 은닉 상태 (자동으로 크기 보정됨)
        """
        # 1. 입력 차원 유연화 (Batch, Seq, Feature)로 통일
        if state.dim() == 2:
            state = state.unsqueeze(1)
        elif state.dim() > 3:
            b = state.size(0)
            d = state.size(-1)
            state = state.view(b, -1, d)

        # 현재 배치의 크기 (Batch Size)
        current_batch_size = state.size(0)

        # 2. [핵심 수정] hidden_state 크기 검사 및 자동 초기화
        if hidden_state is not None:
            # LSTM은 (h, c) 튜플, GRU는 h 텐서
            if self.backbone_type == "lstm":
                h_size = hidden_state[0].size(
                    1
                )  # (Layers, Batch, Hidden) 중 Batch 확인
            else:
                h_size = hidden_state.size(1)

            # 들어온 hidden_state의 배치 크기가 현재 데이터와 다르면 폐기하고 새로 생성
            if h_size != current_batch_size:
                hidden_state = None

        # hidden_state가 없거나(None) 위에서 초기화 대상이 된 경우 새로 생성
        if hidden_state is None:
            hidden_state = self.init_hidden(current_batch_size)

        # RNN 통과
        features, new_hidden_state = self.backbone(state, hidden_state)

        # Multi-Head Output
        target_logits = self.actor_target(features)
        role_logits = self.actor_role(features)
        state_value = self.critic(features)

        return (target_logits, role_logits), state_value, new_hidden_state

    def init_hidden(self, batch_size=1):
        """RNN 은닉 상태 초기화"""
        # 모델 파라미터가 있는 장치(CPU/GPU)를 자동으로 찾음
        device = next(self.parameters()).device

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        if self.backbone_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h, c)
        else:  # gru
            return h
