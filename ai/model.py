import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gru=False, gru_hidden_size=64):
        """
        ActorCritic 모델
        
        Args:
            state_dim: 관측 공간 차원 (168차원)
            action_dim: 액션 공간 차원 (21개)
            use_gru: GRU 레이어 사용 여부 (시계열 추론 강화)
            gru_hidden_size: GRU hidden state 크기
        """
        super(ActorCritic, self).__init__()
        
        self.use_gru = use_gru
        self.gru_hidden_size = gru_hidden_size
        
        # === [기본 MLP 구조: 더 깊은 공통 레이어] ===
        # 시계열 정보를 더 잘 학습하기 위한 깊은 구조
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        if use_gru:
            # === [선택적 GRU 레이어: 시계열 추론 강화] ===
            # GRU를 사용하여 과거 상태를 기억하고 맥락을 유지
            self.gru = nn.GRU(128, gru_hidden_size, batch_first=True)
            self.fc3 = nn.Linear(gru_hidden_size, 64)
        else:
            self.fc3 = nn.Linear(128, 64)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Actor Head (행동 확률)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (가치 판단)
        self.critic = nn.Linear(64, 1)
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, hidden_state=None):
        """
        순방향 전파
        
        Args:
            state: 관측 상태 (batch_size, state_dim) 또는 (batch_size, seq_len, state_dim)
            hidden_state: GRU hidden state (num_layers, batch_size, hidden_size)
            
        Returns:
            action_probs: 행동 확률 분포
            state_value: 상태 가치
            new_hidden_state: 새로운 GRU hidden state (use_gru=True인 경우)
        """
        # 깊은 특징 추출
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        new_hidden_state = None
        if self.use_gru:
            # GRU를 통한 시계열 처리
            # x shape: (batch_size, 128) -> (batch_size, 1, 128)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            if hidden_state is not None:
                x, new_hidden_state = self.gru(x, hidden_state)
            else:
                x, new_hidden_state = self.gru(x)
            
            # x shape: (batch_size, 1, gru_hidden_size) -> (batch_size, gru_hidden_size)
            x = x.squeeze(1)
        
        x = F.relu(self.fc3(x))
        
        # Actor: 행동 확률 분포
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: 상태 가치 평가
        state_value = self.critic(x)
        
        if self.use_gru:
            return action_probs, state_value, new_hidden_state
        else:
            return action_probs, state_value