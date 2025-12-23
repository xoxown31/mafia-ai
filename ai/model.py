import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 공통 레이어 (예시)
        self.fc1 = nn.Linear(state_dim, 64)
        
        # Actor Head (행동 확률)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (가치 판단)
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value