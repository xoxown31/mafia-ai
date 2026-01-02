import torch
import torch.optim as optim
from torch.distributions import Categorical
from ai.model import DynamicActorCritic
from config import config


class REINFORCE:
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.train.LR)
        self.gamma = config.train.GAMMA
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        self.log_probs = []
        self.rewards = []
        self.states = []

    def select_action(self, state, hidden_state=None):
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']
        else:
            obs = state
            mask = None

        state_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
        
        # logits_tuple: (type_logits, target_logits, role_logits)
        logits_tuple, _, new_hidden = self.policy(state_tensor, hidden_state)
        
        target_logits, role_logits = logits_tuple
        target_logits = target_logits.squeeze(0)
        role_logits = role_logits.squeeze(0)
        
        # Masking
        if mask is not None:
            mask_tensor = torch.FloatTensor(mask)
            mask_target = mask_tensor[:9]
            mask_role = mask_tensor[9:]
            
            if mask_target.sum() > 0:
                target_logits = target_logits.masked_fill(mask_target == 0, -1e9)
            if mask_role.sum() > 0:
                role_logits = role_logits.masked_fill(mask_role == 0, -1e9)
        
        dist_target = Categorical(logits=target_logits)
        dist_role = Categorical(logits=role_logits)
        
        action_target = dist_target.sample()
        action_role = dist_role.sample()
        
        log_prob = dist_target.log_prob(action_target) + dist_role.log_prob(action_role)
        
        action = [action_target.item(), action_role.item()]
        
        self.log_probs.append(log_prob)
        self.states.append(state_tensor.squeeze(0))
        
        return action, new_hidden

    def update(self, il_loss_fn=None):
        if len(self.rewards) == 0:
            return
            
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
            
        loss = torch.stack(policy_loss).mean()
        
        if il_loss_fn is not None:
            old_states = torch.stack(self.states, dim=0).detach()
            il_loss = il_loss_fn(old_states)
            loss = loss + config.train.IL_COEF * il_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.clear_memory()

    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.states[:]