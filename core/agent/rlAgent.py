import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from torch.distributions import Categorical

from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent
from ai.model import DynamicActorCritic
from ai.ppo import PPO
from ai.reinforce import REINFORCE
from config import config, Role
from state import GameStatus, GameEvent, GameAction


class RLAgent(BaseAgent):
    """
    PPO, REINFORCE, IL, RNN을 모두 지원하는 통합 RL 에이전트
    
    Multi-Discrete 액션 공간 지원: [Target, Role]
    
    Args:
        player_id: 플레이어 ID
        role: 플레이어 역할
        state_dim: 상태 벡터 차원
        action_dims: Multi-Discrete 액션 차원 리스트 [9, 5]
        algorithm: "ppo" 또는 "reinforce"
        backbone: "lstm", "gru"
        use_il: Imitation Learning 사용 여부
        hidden_dim: 은닉층 차원
        num_layers: RNN 레이어 수
    """

    def __init__(
        self,
        player_id: int,
        role: Role,
        state_dim: int,
        action_dims: List[int] = [9, 5],  # [Target, Role]
        algorithm: str = "ppo",
        backbone: str = "lstm",
        use_il: bool = False,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__(player_id, role)

        self.algorithm = algorithm.lower()
        self.backbone = backbone.lower()
        self.use_il = use_il
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.action_dim = sum(action_dims)
        
        self.policy = DynamicActorCritic(
            state_dim=state_dim,
            action_dims=action_dims,
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        if self.algorithm == "ppo":
            self.learner = PPO(policy=self.policy)
        elif self.algorithm == "reinforce":
            self.learner = REINFORCE(policy=self.policy)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.hidden_state = None
        self.current_action = None  # 현재 액션 저장

        self.expert_agent = None
        if use_il:
            self.expert_agent = LLMAgent(player_id, role)
            self.il_buffer = []

    def reset_hidden(self):
        """에피소드 시작 시 RNN 은닉 상태 초기화"""
        if self.backbone in ["lstm", "gru"]:
            self.hidden_state = self.policy.init_hidden(batch_size=1)
        else:
            self.hidden_state = None

    def update_belief(self, history: List[GameEvent]):
        """BaseAgent의 추상 메서드 구현"""
        pass

    def set_action(self, action: GameAction):
        """env.step()에서 호출되어 액션 설정"""
        self.current_action = action

    def get_action(self) -> GameAction:
        """
        BaseAgent 호환성을 위한 메서드 - MafiaAction 반환

        env.step()에서 설정한 current_action을 반환
        """
        if self.current_action is not None:
            return self.current_action

        # 폴백: PASS 액션
        return GameAction(target_id=-1, claim_role=None)
    
    def select_action_vector(self, state, action_mask: Optional[np.ndarray] = None) -> List[int]:
        """
        상태를 받아 Multi-Discrete 액션 벡터를 선택

        Args:
            state: 상태 벡터 또는 딕셔너리
            action_mask: 유효한 행동 마스크 (9, 5) 형태

        Returns:
            action_vector: [Target, Role] 형태의 리스트
        """
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask", action_mask)
        else:
            obs = state
            mask = action_mask
        
        state_dict = {'observation': obs, 'action_mask': mask}
        
        # learner.select_action returns ([target, role], hidden_state)
        action_vector, self.hidden_state = self.learner.select_action(state_dict, self.hidden_state)
        
        return action_vector
    
    def store_reward(self, reward: float, is_terminal: bool = False):
        """보상 저장: 알고리즘별 버퍼 위치 확인"""
        if hasattr(self.learner, 'buffer'):
            # PPO의 경우 buffer 객체 내의 리스트 사용
            self.learner.buffer.rewards.append(reward)
            self.learner.buffer.is_terminals.append(is_terminal)
        else:
            # REINFORCE 등 buffer가 없는 경우 직접 저장
            self.learner.rewards.append(reward)
    
    def update(self):
        """학습 수행 - 알고리즘 객체에 위임"""
        if self.use_il and self.expert_agent is not None:
            il_loss_fn = self._create_il_loss_fn()
            self.learner.update(il_loss_fn=il_loss_fn)
        else:
            self.learner.update()

    def _create_il_loss_fn(self):
        """IL 손실 함수 생성 (최적화됨)"""

        def compute_il_loss(states: torch.Tensor) -> torch.Tensor:
            if len(self.il_buffer) == 0:
                return torch.tensor(0.0)

            # 텐서 생성 최적화: 리스트 컴프리헨션 대신 스택 사용
            il_states_list = [s for s, _, _ in self.il_buffer]
            il_actions_list = [a for _, a, _ in self.il_buffer]

            if len(il_states_list) == 0:
                return torch.tensor(0.0)

            # 기존 텐서로 변환
            il_states = torch.stack(
                [torch.as_tensor(s, dtype=torch.float32) for s in il_states_list]
            )
            il_actions = torch.tensor(il_actions_list, dtype=torch.long)

            # RNN 처리: 배치 유지
            if self.backbone in ["lstm", "gru"]:
                # 모든 샘플을 하나의 시퀀스로 처리
                il_states = il_states.unsqueeze(0)
                action_probs, _, _ = self.policy(il_states)
                action_probs = action_probs.squeeze(0)
            else:
                action_probs, _, _ = self.policy(il_states)

            # 마스크 처리 (선택적)
            # 버퍼에 마스크 정보가 있다면 적용

            criterion = nn.CrossEntropyLoss()
            loss = criterion(action_probs, il_actions)

            return loss

        return compute_il_loss

    def collect_expert_experience(self, state: np.ndarray, action_mask: np.ndarray):
        """전문가(LLM) 행동 수집"""
        if not self.use_il or self.expert_agent is None:
            return None

        self.expert_agent.observe(self.current_status)
        expert_action_str = self.expert_agent.get_action()

        try:
            expert_action_data = json.loads(expert_action_str)
            if "target_id" in expert_action_data:
                expert_action = expert_action_data["target_id"]
            elif "action" in expert_action_data:
                expert_action = expert_action_data["action"]
            else:
                return None

            if 0 <= expert_action < self.action_dim and action_mask[expert_action] == 1:
                self.il_buffer.append((state, expert_action, action_mask))
                if len(self.il_buffer) > 1000:
                    self.il_buffer.pop(0)
                return expert_action
        except:
            pass

        return None

    def pretrain_il(
        self,
        expert_trajectories: List[Tuple[np.ndarray, int, np.ndarray]],
        num_epochs: int = 10,
    ):
        """
        LLM 에이전트의 행동을 모방하는 사전 학습

        Args:
            expert_trajectories: [(state, action, mask), ...]
            num_epochs: 사전 학습 에포크 수
        """
        if not self.use_il:
            print("IL is not enabled for this agent")
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.train.LR)

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

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.train.MAX_GRAD_NORM
                )
                optimizer.step()

                total_loss += loss.item()
                pred = action_probs.argmax(dim=-1)
                correct += (pred == expert_action_tensor).sum().item()
                total += 1

            avg_loss = total_loss / len(expert_trajectories)
            accuracy = 100.0 * correct / total

            if (epoch + 1) % 5 == 0:
                print(
                    f"IL Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )

        if self.algorithm == "ppo":
            self.learner.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filepath: str):
        """모델 저장"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "algorithm": self.algorithm,
                "backbone": self.backbone,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            filepath,
        )

    def load(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])

        if self.algorithm == "ppo":
            self.learner.policy_old.load_state_dict(self.policy.state_dict())
