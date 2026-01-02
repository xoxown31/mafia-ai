import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicActorCritic(nn.Module):
    """
    RNN(LSTM/GRU) 기반의 Actor-Critic 모델
    
    Args:
        state_dim: 입력 상태 차원 (78)
        action_dims: 출력 행동 차원 리스트 (예: [9, 5])
        backbone: 백본 타입 ("lstm", "gru")
        hidden_dim: 은닉층 차원 (기본값: 128)
        num_layers: RNN 레이어 수 (기본값: 2)
    """
    def __init__(self, state_dim, action_dims=[9, 5], backbone="lstm", hidden_dim=128, num_layers=2):
        super(DynamicActorCritic, self).__init__()
        
        self.backbone_type = backbone.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        if self.backbone_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
        elif self.backbone_type == "gru":
            self.backbone = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'lstm' or 'gru'")
        
        feature_dim = hidden_dim
        
        # Multi-Discrete Action Heads
        # Action Space: [Target(9), Role(5)]
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
            state: 입력 상태 [batch_size, seq_len, state_dim]
            hidden_state: RNN 은닉 상태
        
        Returns:
            action_logits: (target_logits, role_logits) 튜플
            state_value: 상태 가치 [batch_size, 1]
            new_hidden_state: 새로운 은닉 상태
        """
        # RNN은 3D 입력 필요: [batch, seq_len, features]
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        if hidden_state is not None:
            features, new_hidden_state = self.backbone(state, hidden_state)
        else:
            features, new_hidden_state = self.backbone(state)
        
        # features: [batch, seq_len, hidden_dim]
        # Apply heads to all timesteps
        
        # Multi-Head Output
        target_logits = self.actor_target(features)
        role_logits = self.actor_role(features)
        
        state_value = self.critic(features)
        
        return (target_logits, role_logits), state_value, new_hidden_state
    
    def init_hidden(self, batch_size=1):
        """RNN 은닉 상태 초기화"""
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        
        if self.backbone_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h, c)
        else:  # gru
            return h
