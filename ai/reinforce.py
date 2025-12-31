import torch
import torch.optim as optim
from torch.distributions import Categorical
from ai.model import ActorCritic
from config import config


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.train.LR)
        self.gamma = config.train.GAMMA
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']
        else:
            obs = state
            mask = None

        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0) if mask is not None else None
        
        probs, _, _ = self.policy(state_tensor)
        
        if mask_tensor is not None:
            probs = probs * mask_tensor
            total_prob = probs.sum(dim=-1, keepdim=True)
            probs = probs / (total_prob + 1e-8)
        
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()

    def update(self):
        R = 0
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.clear_memory()

    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]