import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from torch.distributions import Categorical

from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent
from ai.model import DynamicActorCritic
from ai.buffer import RolloutBuffer
from config import config, Role
from state import GameStatus, GameEvent


class RLAgent(BaseAgent):
    """
    PPO, REINFORCE, IL(Imitation Learning), RNN을 모두 지원하는 통합 RL 에이전트
    
    Args:
        player_id: 플레이어 ID
        role: 플레이어 역할
        state_dim: 상태 벡터 차원
        action_dim: 행동 공간 크기
        algorithm: "ppo" 또는 "reinforce"
        backbone: "mlp", "lstm", "gru"
        use_il: Imitation Learning 사용 여부
        hidden_dim: 은닉층 차원
        num_layers: RNN 레이어 수
    """
    
    def __init__(
        self,
        player_id: int,
        role: Role,
        state_dim: int,
        action_dim: int,
        algorithm: str = "ppo",
        backbone: str = "mlp",
        use_il: bool = False,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__(player_id, role)
        
        self.algorithm = algorithm.lower()
        self.backbone = backbone.lower()
        self.use_il = use_il
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy = DynamicActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.train.LR)
        
        if self.algorithm == "ppo":
            self.policy_old = DynamicActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                backbone=backbone,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            self.policy_old.load_state_dict(self.policy.state_dict())
            self.buffer = RolloutBuffer()
            self.eps_clip = config.train.EPS_CLIP
            self.k_epochs = config.train.K_EPOCHS
        else:
            self.log_probs = []
            self.rewards = []
        
        self.gamma = config.train.GAMMA
        self.entropy_coef = config.train.ENTROPY_COEF
        self.value_loss_coef = config.train.VALUE_LOSS_COEF
        self.max_grad_norm = config.train.MAX_GRAD_NORM
        self.il_coef = config.train.IL_COEF if use_il else 0.0
        
        self.hidden_state = None
        self.MseLoss = nn.MSELoss()
        
        self.expert_agent = None
        if use_il:
            self.expert_agent = LLMAgent(player_id, role)
    
    def reset_hidden(self):
        """에피소드 시작 시 RNN 은닉 상태 초기화"""
        if self.backbone in ["lstm", "gru"]:
            self.hidden_state = self.policy.init_hidden(batch_size=1)
        else:
            self.hidden_state = None
    
    def update_belief(self, history: List[GameEvent]):
        """BaseAgent의 추상 메서드 구현"""
        for event in history:
            pass
    
    def get_action(self) -> str:
        """BaseAgent의 추상 메서드 구현 - RL 에이전트는 select_action 사용"""
        return json.dumps({"error": "Use select_action method for RL agents"})
    
    def select_action(self, state, action_mask: Optional[np.ndarray] = None):
        """
        상태를 받아 행동을 선택하고 버퍼에 저장
        
        Args:
            state: 상태 벡터 또는 딕셔너리 {"observation": ..., "action_mask": ...}
            action_mask: 유효한 행동 마스크 (1: 가능, 0: 불가능)
        
        Returns:
            action: 선택된 행동 인덱스
        """
        if isinstance(state, dict):
            obs = state['observation']
            mask = state.get('action_mask', action_mask)
        else:
            obs = state
            mask = action_mask
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            if self.algorithm == "ppo":
                policy_net = self.policy_old
            else:
                policy_net = self.policy
            
            action_probs, state_value, new_hidden = policy_net(
                state_tensor, 
                self.hidden_state
            )
            
            if self.backbone in ["lstm", "gru"]:
                self.hidden_state = new_hidden
            
            action_probs = action_probs.squeeze(0)
            
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                action_probs = action_probs * mask_tensor
                prob_sum = action_probs.sum()
                if prob_sum > 0:
                    action_probs = action_probs / prob_sum
                else:
                    action_probs = mask_tensor / mask_tensor.sum()
            
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        if self.algorithm == "ppo":
            self.buffer.states.append(state_tensor.squeeze(0))
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            if self.backbone in ["lstm", "gru"]:
                self.buffer.hidden_states.append(self.hidden_state)
        else:
            self.log_probs.append(action_logprob)
        
        return action.item()
    
    def store_reward(self, reward: float, is_terminal: bool = False):
        """보상 저장"""
        if self.algorithm == "ppo":
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(is_terminal)
        else:
            self.rewards.append(reward)
    
    def update(self):
        """학습 수행"""
        if self.algorithm == "ppo":
            self._update_ppo()
        else:
            self._update_reinforce()
    
    def _update_ppo(self):
        """PPO 업데이트"""
        if len(self.buffer.rewards) == 0:
            return
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals)
        ):
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
            
            loss = (
                actor_loss.mean() + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss.mean()
            )
            
            if self.use_il and self.expert_agent is not None:
                il_loss = self._compute_il_loss(old_states)
                loss = loss + self.il_coef * il_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def _update_reinforce(self):
        """REINFORCE 업데이트"""
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
        
        if self.use_il and self.expert_agent is not None:
            il_loss = self._compute_il_loss(None)
            loss = loss + self.il_coef * il_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.clear_memory()
    
    def clear_memory(self):
        """REINFORCE 메모리 초기화"""
        if self.algorithm == "reinforce":
            del self.log_probs[:]
            del self.rewards[:]
    
    def _compute_il_loss(self, states: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Imitation Learning 손실 계산 (Behavior Cloning)
        
        Args:
            states: 상태 텐서 (PPO의 경우), None (REINFORCE의 경우)
        
        Returns:
            il_loss: IL 손실값
        """
        if not self.use_il or self.expert_agent is None:
            return torch.tensor(0.0)
        
        return torch.tensor(0.0)
    
    def pretrain_il(
        self,
        expert_trajectories: List[Tuple[np.ndarray, int, np.ndarray]],
        num_epochs: int = 10
    ):
        """
        LLM 에이전트의 행동을 모방하는 사전 학습
        
        Args:
            expert_trajectories: [(state, action, mask), ...] 형태의 전문가 데이터
            num_epochs: 사전 학습 에포크 수
        """
        if not self.use_il:
            print("IL is not enabled for this agent")
            return
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for state, expert_action, mask in expert_trajectories:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                expert_action_tensor = torch.LongTensor([expert_action])
                
                action_probs, _, _ = self.policy(state_tensor)
                
                if mask is not None:
                    mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
                    action_probs = action_probs * mask_tensor
                    prob_sum = action_probs.sum(dim=-1, keepdim=True)
                    action_probs = action_probs / (prob_sum + 1e-8)
                
                loss = criterion(action_probs, expert_action_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                pred = action_probs.argmax(dim=-1)
                correct += (pred == expert_action_tensor).sum().item()
                total += 1
            
            avg_loss = total_loss / len(expert_trajectories)
            accuracy = 100.0 * correct / total
            
            if (epoch + 1) % 5 == 0:
                print(f"IL Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if self.algorithm == "ppo":
            self.policy_old.load_state_dict(self.policy.state_dict())
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'algorithm': self.algorithm,
            'backbone': self.backbone,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
    
    def load(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.algorithm == "ppo":
            self.policy_old.load_state_dict(self.policy.state_dict())
