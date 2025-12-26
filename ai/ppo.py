import torch
import torch.nn as nn
from torch.distributions import Categorical
import config
from ai.model import ActorCritic
from ai.buffer import RolloutBuffer

class PPO:
    def __init__(self, state_dim, action_dim, use_gru=False):
        """
        PPO 알고리즘
        
        Args:
            state_dim: 관측 공간 차원
            action_dim: 액션 공간 차원
            use_gru: GRU 사용 여부 (시계열 추론 강화)
        """
        self.gamma = config.GAMMA
        self.eps_clip = config.EPS_CLIP
        self.k_epochs = config.K_EPOCHS
        self.lr = config.LR
        
        # 새로운 하이퍼파라미터
        self.entropy_coef = config.ENTROPY_COEF
        self.value_loss_coef = config.VALUE_LOSS_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        
        # GRU 사용 여부
        self.use_gru = use_gru
        
        # 데이터 수집을 위한 버퍼 생성
        self.buffer = RolloutBuffer()
        
        # 현재 정책 (학습 대상)
        self.policy = ActorCritic(state_dim, action_dim, use_gru=use_gru)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 이전 정책 (Ratio 계산용, 가중치 고정)
        self.policy_old = ActorCritic(state_dim, action_dim, use_gru=use_gru)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # GRU hidden state (에피소드 중 유지)
        self.hidden_state = None

    def select_action(self, state, reset_hidden=False):
        """
        상태(Dict)를 받아 마스킹을 적용한 후 행동을 결정
        
        Args:
            state: 관측 상태 (dict 형태: {'observation': ..., 'action_mask': ...})
            reset_hidden: GRU hidden state 초기화 여부 (에피소드 시작 시)
        """
        # 에피소드 시작 시 hidden state 초기화
        if reset_hidden:
            self.hidden_state = None
        
        # 1. Dict에서 관측값과 마스크 분리
        if isinstance(state, dict):
            obs = state['observation']
            mask = state['action_mask']
        else:
            obs = state
            mask = None
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask) if mask is not None else None
            
            if self.use_gru:
                action_probs, _, new_hidden = self.policy_old(state_tensor, self.hidden_state)
                self.hidden_state = new_hidden
                action_probs = action_probs.squeeze(0)
            else:
                action_probs, _ = self.policy_old(state_tensor)
                action_probs = action_probs.squeeze(0)
            
            # 2. 마스킹 적용
            if mask_tensor is not None:
                # 불가능한 행동(0)의 확률을 0으로 만듦
                action_probs = action_probs * mask_tensor
                
                # 확률 재정규화 (Sum to 1)
                action_probs_sum = action_probs.sum()
                if action_probs_sum > 0:
                    action_probs /= action_probs_sum
                else:
                    # 예외 처리: 모든 행동이 불가능한 경우(버그 등) 균등 분포 등 처리 필요
                    print("Warning: All actions masked out!")
                    action_probs = mask_tensor / mask_tensor.sum()

            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        self.buffer.states.append(state_tensor.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.item()

    def update(self):
        """
        모인 데이터를 바탕으로 PPO 업데이트 수행
        """
        # 1. Monte Carlo Estimate of Returns (Discounted Reward 계산)
        rewards = []
        discounted_reward = 0
        # 버퍼의 뒤에서부터 보상을 누적 계산
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 텐서 변환 및 정규화 (학습 안정성)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 버퍼에 있는 데이터를 텐서로 변환 (detach로 그래프 끊기)
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        
        for _ in range(self.k_epochs):
            if self.use_gru:
                action_probs, state_values, _ = self.policy(old_states, None)
            else:
                action_probs, state_values = self.policy(old_states)
            
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = torch.squeeze(state_values)
            
            # Ratio 계산 (pi_theta / pi_theta_old)
            # log_prob끼리의 뺄셈 후 exp는 나눗셈과 같음
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss 계산
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 최종 Loss: Actor Loss + Value Loss - Entropy Bonus
            # config에서 정의한 계수 사용
            actor_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values, rewards)
            entropy_loss = -dist_entropy
            
            loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient Clipping (학습 안정성 향상)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
        # 3. 학습된 정책을 Old Policy로 복사
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 4. 버퍼 비우기
        self.buffer.clear()