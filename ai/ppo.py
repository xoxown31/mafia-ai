import torch
import torch.nn as nn
from torch.distributions import Categorical
from config import config
from ai.model import DynamicActorCritic
from ai.buffer import RolloutBuffer


class PPO:
    def __init__(self, policy, policy_old=None):
        self.gamma = config.train.GAMMA
        self.eps_clip = config.train.EPS_CLIP
        self.k_epochs = config.train.K_EPOCHS
        self.lr = config.train.LR
        self.entropy_coef = config.train.ENTROPY_COEF
        self.value_loss_coef = config.train.VALUE_LOSS_COEF
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        
        self.buffer = RolloutBuffer()
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        if policy_old is None:
            self.policy_old = DynamicActorCritic(
                state_dim=policy.state_dim,
                action_dim=policy.actor.out_features,
                backbone=policy.backbone_type,
                hidden_dim=policy.hidden_dim,
                num_layers=policy.num_layers
            )
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy_old = policy_old
            
        self.MseLoss = nn.MSELoss()
        self.is_rnn = self.policy.backbone_type in ["lstm", "gru"]

    def select_action(self, state, hidden_state=None):
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']
        else:
            obs = state
            mask = None
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask) if mask is not None else None
            
            action_probs, _, new_hidden = self.policy_old(state_tensor, hidden_state)
            action_probs = action_probs.squeeze(0)
            
            if mask_tensor is not None:
                action_probs = action_probs * mask_tensor
                action_probs_sum = action_probs.sum()
                if action_probs_sum > 0:
                    action_probs /= action_probs_sum
                else:
                    print("Warning: All actions masked out!")
                    action_probs = mask_tensor / mask_tensor.sum()

            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        self.buffer.states.append(state_tensor.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        if self.is_rnn and new_hidden is not None:
            self.buffer.hidden_states.append(new_hidden)
        
        return action.item(), new_hidden

    def update(self, il_loss_fn=None):
        if len(self.buffer.rewards) == 0:
            return
            
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        
        for _ in range(self.k_epochs):
            if self.is_rnn:
                # RNN: 에피소드 경계를 고려한 시퀀스 처리
                episodes = self._split_episodes(old_states, self.buffer.is_terminals)
                all_action_probs = []
                all_state_values = []
                
                for ep_states in episodes:
                    if len(ep_states) == 0:
                        continue
                    ep_states_3d = ep_states.unsqueeze(0)
                    action_probs, state_values, _ = self.policy(ep_states_3d)
                    all_action_probs.append(action_probs.squeeze(0))
                    all_state_values.append(state_values.squeeze(0))
                
                action_probs = torch.cat(all_action_probs, dim=0)
                state_values = torch.cat(all_state_values, dim=0)
            else:
                action_probs, state_values, _ = self.policy(old_states)
            
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values, rewards)
            entropy_loss = -dist_entropy
            
            loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            if il_loss_fn is not None:
                il_loss = il_loss_fn(old_states)
                loss = loss + config.train.IL_COEF * il_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def _split_episodes(self, states, is_terminals):
        """에피소드 경계에 따라 상태를 분리"""
        episodes = []
        current_episode = []
        
        for i, state in enumerate(states):
            current_episode.append(state)
            if is_terminals[i]:
                if current_episode:
                    episodes.append(torch.stack(current_episode))
                current_episode = []
        
        if current_episode:
            episodes.append(torch.stack(current_episode))
        
        return episodes