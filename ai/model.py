import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicActorCritic(nn.Module):
    """
    다양한 백본(MLP, LSTM, GRU)을 지원하는 Actor-Critic 모델
    
    Args:
        state_dim: 입력 상태 차원
        action_dim: 출력 행동 차원
        backbone: 백본 타입 ("mlp", "lstm", "gru")
        hidden_dim: 은닉층 차원 (기본값: 128)
        num_layers: RNN 레이어 수 (기본값: 2)
    """
    def __init__(self, state_dim, action_dim, backbone="mlp", hidden_dim=128, num_layers=2):
        super(DynamicActorCritic, self).__init__()
        
        self.backbone_type = backbone.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        
        if self.backbone_type == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 64),
                nn.ReLU()
            )
            feature_dim = 64
            
        elif self.backbone_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            feature_dim = hidden_dim
            
        elif self.backbone_type == "gru":
            self.backbone = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            feature_dim = hidden_dim
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'mlp', 'lstm', or 'gru'")
        
        self.actor = nn.Linear(feature_dim, action_dim)
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
            state: 입력 상태 [batch_size, state_dim] 또는 [batch_size, seq_len, state_dim]
            hidden_state: RNN 은닉 상태 (LSTM의 경우 (h, c), GRU의 경우 h)
        
        Returns:
            action_probs: 행동 확률 분포 [batch_size, action_dim]
            state_value: 상태 가치 [batch_size, 1]
            new_hidden_state: 새로운 은닉 상태 (RNN의 경우)
        """
        if self.backbone_type == "mlp":
            features = self.backbone(state)
            new_hidden_state = None
            
        elif self.backbone_type in ["lstm", "gru"]:
            # RNN은 3D 입력 필요: [batch, seq_len, features]
            if state.dim() == 2:
                state = state.unsqueeze(1)
            
            if hidden_state is not None:
                features, new_hidden_state = self.backbone(state, hidden_state)
            else:
                features, new_hidden_state = self.backbone(state)
            
            # 마지막 타임스텝의 출력 사용
            features = features[:, -1, :]
        
        action_probs = F.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        
        return action_probs, state_value, new_hidden_state
    
    def init_hidden(self, batch_size=1):
        """RNN 은닉 상태 초기화"""
        if self.backbone_type == "mlp":
            return None
        
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        
        if self.backbone_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h, c)
        else:  # gru
            return h


# 레거시 호환성을 위한 별칭
ActorCritic = DynamicActorCritic