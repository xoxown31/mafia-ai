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
            # DynamicActorCritic ignores action_dim for heads (hardcoded 9, 5)
            # So we can pass any value or the sum
            self.policy_old = DynamicActorCritic(
                state_dim=policy.state_dim,
                action_dims=policy.action_dims, 
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
            mask = state['action_mask']  # Shape: (14,)
        else:
            obs = state
            mask = None
            
        with torch.no_grad():
            # obs: (78,) -> (1, 1, 78) for RNN (batch, seq, feature)
            # or (1, 78) for MLP
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if self.is_rnn:
                state_tensor = state_tensor.unsqueeze(0)
            
            # logits_tuple: (target_logits, role_logits)
            logits_tuple, _, new_hidden = self.policy_old(state_tensor, hidden_state)
            
            # Unpack logits
            target_logits, role_logits = logits_tuple
            target_logits = target_logits.squeeze(0)
            role_logits = role_logits.squeeze(0)
            
            # Apply Masking if available
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                # mask shape: (14,) -> [Target(9), Role(5)]
                mask_target = mask_tensor[:9]
                mask_role = mask_tensor[9:]
                
                # Safety check: if all masked, ignore mask
                if mask_target.sum() > 0:
                    target_logits = target_logits.masked_fill(mask_target == 0, -1e9)
                if mask_role.sum() > 0:
                    role_logits = role_logits.masked_fill(mask_role == 0, -1e9)

            # Create distributions
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            # Sample actions
            action_target = dist_target.sample()
            action_role = dist_role.sample()
            
            # Calculate log probs
            logprob_target = dist_target.log_prob(action_target)
            logprob_role = dist_role.log_prob(action_role)
            
            # Total log prob (sum of independent log probs)
            action_logprob = logprob_target + logprob_role
            
            action = torch.stack([action_target, action_role])
            
        # Store in buffer
        # For RNN, we store the state as (78,) and reconstruct sequences in update()
        self.buffer.states.append(torch.FloatTensor(obs))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        # Note: We don't store hidden_states in buffer for PPO update usually, 
        # because we re-run the policy on the trajectory.
        # But if we want to support truncated BPTT, we might need them.
        # Here we assume full episode training or simple batching.
        
        return action.tolist(), new_hidden

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
        if rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach() # Shape: (N, 2)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        
        for _ in range(self.k_epochs):
            if self.is_rnn:
                # RNN: 에피소드 경계를 고려한 시퀀스 처리
                ep_states_list = self._split_episodes(self.buffer.states, self.buffer.is_terminals)
                ep_actions_list = self._split_episodes(self.buffer.actions, self.buffer.is_terminals)
                ep_logprobs_list = self._split_episodes(self.buffer.logprobs, self.buffer.is_terminals)
                ep_rewards_list = self._split_episodes(rewards, self.buffer.is_terminals)
                
                total_loss = 0
                for i in range(len(ep_states_list)):
                    # (seq_len, dim) -> (1, seq_len, dim)
                    ep_states = ep_states_list[i].unsqueeze(0)
                    ep_actions = ep_actions_list[i]
                    ep_old_logprobs = ep_logprobs_list[i]
                    ep_rewards = ep_rewards_list[i]
                    
                    # Forward pass (hidden state is reset for each episode)
                    logits_tuple, state_values, _ = self.policy(ep_states)
                    target_logits, role_logits = logits_tuple
                    
                    # Remove batch dim: (1, seq, dim) -> (seq, dim)
                    target_logits = target_logits.squeeze(0)
                    role_logits = role_logits.squeeze(0)
                    state_values = state_values.squeeze(0).squeeze(-1)
                    
                    dist_target = Categorical(logits=target_logits)
                    dist_role = Categorical(logits=role_logits)
                    
                    logprobs_target = dist_target.log_prob(ep_actions[:, 0])
                    logprobs_role = dist_role.log_prob(ep_actions[:, 1])
                    logprobs = logprobs_target + logprobs_role
                    
                    dist_entropy = dist_target.entropy() + dist_role.entropy()
                    
                    ratios = torch.exp(logprobs - ep_old_logprobs)
                    
                    advantages = ep_rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    
                    loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, ep_rewards) - self.entropy_coef * dist_entropy
                    total_loss += loss.mean()
                
                loss = total_loss / len(ep_states_list)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                continue

            # Forward pass
            logits_tuple, state_values, _ = self.policy(old_states)
            target_logits, role_logits = logits_tuple
            
            # Create distributions
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)
            
            # Get log probs for old actions
            # old_actions: (N, 2) -> target, role
            logprobs_target = dist_target.log_prob(old_actions[:, 0])
            logprobs_role = dist_role.log_prob(old_actions[:, 1])
            
            logprobs = logprobs_target + logprobs_role
            dist_entropy = dist_target.entropy() + dist_role.entropy()
            state_values = state_values.squeeze()
            
            # Ratios
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # IL Loss
            if il_loss_fn is not None:
                loss = loss.mean() + config.train.IL_COEF * il_loss_fn(old_states)
            else:
                loss = loss.mean()
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def _split_episodes(self, data_list, is_terminals):
        """에피소드 경계에 따라 데이터를 분리"""
        episodes = []
        current_episode = []
        
        for i, data in enumerate(data_list):
            current_episode.append(data)
            if is_terminals[i]:
                if current_episode:
                    episodes.append(torch.stack(current_episode))
                current_episode = []
        
        # 마지막 에피소드가 끝나지 않았더라도 추가 (Truncated)
        if current_episode:
            episodes.append(torch.stack(current_episode))
        
        return episodes