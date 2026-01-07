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
        self.is_rnn = policy.backbone_type in ["lstm", "gru"]

        if policy_old is None:
            self.policy_old = DynamicActorCritic(
                state_dim=policy.state_dim,
                action_dims=policy.action_dims,
                backbone=policy.backbone_type,
                hidden_dim=policy.hidden_dim,
                num_layers=policy.num_layers,
            )
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy_old = policy_old

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden_state=None):
        """
        상태를 입력받아 행동을 결정합니다. (배치 처리 지원)
        """
        if isinstance(state, dict):
            obs = state["observation"]
            mask = state.get("action_mask")
        else:
            obs = state
            mask = None

        with torch.no_grad():
            # [수정] 입력 차원에 따른 유동적인 배치 처리
            state_tensor = torch.FloatTensor(obs)

            # Case 1: 단일 입력 (Dim,) -> (1, 1, Dim)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            # Case 2: 배치 입력 (Batch, Dim) -> (Batch, 1, Dim)
            elif state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(1)

            # 모델 실행
            logits_tuple, _, new_hidden = self.policy_old(state_tensor, hidden_state)

            # 결과 분리
            target_logits, role_logits = logits_tuple

            # (Batch, 1, Dim) -> (Batch, Dim) or (Dim,)
            if state_tensor.size(0) > 1:
                target_logits = target_logits.squeeze(1)  # (Batch, 9)
                role_logits = role_logits.squeeze(1)  # (Batch, 5)
            else:
                target_logits = target_logits.view(-1)
                role_logits = role_logits.view(-1)

            # 마스킹 적용 (Batch 지원)
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)  # (1, 14)

                # 마스크 분리 [Target(9) | Role(5)]
                mask_target = mask_tensor[:, :9]
                mask_role = mask_tensor[:, 9:]

                # 마스킹 연산 (Broadcasting)
                # 안전장치: 모든 액션이 마스킹되는 경우 방지 (sum > 0 일때만 적용)
                valid_target = mask_target.sum(dim=1, keepdim=True) > 0
                valid_role = mask_role.sum(dim=1, keepdim=True) > 0

                target_logits = target_logits.masked_fill(
                    (mask_target == 0) & valid_target, -1e9
                )
                role_logits = role_logits.masked_fill(
                    (mask_role == 0) & valid_role, -1e9
                )

            # 확률 분포 생성 및 샘플링
            dist_target = Categorical(logits=target_logits)
            dist_role = Categorical(logits=role_logits)

            action_target = dist_target.sample()
            action_role = dist_role.sample()

            # 로그 확률 계산
            logprob_target = dist_target.log_prob(action_target)
            logprob_role = dist_role.log_prob(action_role)
            action_logprob = logprob_target + logprob_role

            # 행동 스택 (Batch, 2)
            action = torch.stack([action_target, action_role], dim=-1)

        # 버퍼 저장 (Batch 단위로 저장하지 않고, 리스트에 텐서를 추가하는 방식 유지)
        # 주의: PPO 업데이트 시 텐서들을 stack하므로 여기서는 텐서 그대로 저장
        self.buffer.states.append(torch.FloatTensor(obs))  # (Batch, Dim) 저장
        self.buffer.actions.append(action)  # (Batch, 2) 저장
        self.buffer.logprobs.append(action_logprob)  # (Batch,) 저장

        # 병렬 환경일 경우 action을 list로 변환하여 반환
        return action.tolist(), new_hidden

    def update(self, il_loss_fn=None):
        if len(self.buffer.rewards) == 0:
            return {}

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        if len(rewards) > 1 and rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 병렬 환경에서는 buffer에 저장된 요소들이 (Batch, ...) 형태일 수 있음
        # 이를 (Time * Batch, ...) 형태로 Flatten 하거나 (Time, Batch, ...) 유지
        # 여기서는 간단하게 Stack 후 차원 정리

        # List[(Batch, Dim)] -> Tensor(Time, Batch, Dim)
        try:
            old_states = torch.stack(self.buffer.states, dim=0).detach()
            old_actions = torch.stack(self.buffer.actions, dim=0).detach()
            old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        except RuntimeError:
            # 혹시 차원이 안 맞으면 cat으로 시도 (가변 길이 방지)
            old_states = torch.cat(
                [
                    s.unsqueeze(0) if s.dim() == 1 else s.unsqueeze(0)
                    for s in self.buffer.states
                ],
                dim=0,
            ).detach()
            old_actions = torch.cat(
                [
                    a.unsqueeze(0) if a.dim() == 1 else a.unsqueeze(0)
                    for a in self.buffer.actions
                ],
                dim=0,
            ).detach()
            old_logprobs = torch.cat(
                [
                    l.unsqueeze(0) if l.dim() == 0 else l.unsqueeze(0)
                    for l in self.buffer.logprobs
                ],
                dim=0,
            ).detach()

        # 30000판 처리를 위해 데이터를 단순히 플래튼(Flatten)하여 학습 (RNN 시퀀스 무시)
        # 또는 Batch 차원을 유지하려면 복잡한 처리가 필요함.
        # 여기서는 가장 안정적인 방법인 Flatten 적용 (Batch와 Time을 합침)
        if old_states.dim() == 3:  # (Time, Batch, Dim)
            old_states = old_states.view(-1, old_states.size(-1))
            old_actions = old_actions.view(-1, old_actions.size(-1))
            old_logprobs = old_logprobs.view(-1)
            rewards = rewards.view(
                -1
            )  # Reward도 (Time, Batch)라면 Flatten 필요할 수 있음
            # 주의: Reward는 이미 1D 리스트로 들어오므로 (Time * Batch) 길이일 것임 (Runner 구현에 따라 다름)
            # Runner가 reward를 extend 했으면 1D, append 했으면 문제됨.
            # 현재 Runner는 store_reward 루프를 돌므로 1D로 평탄화되어 들어옴.

            # 하지만 buffer.states는 append(Tensor(Batch, Dim)) 했으므로 Stack하면 (Time, Batch, Dim).
            # rewards는 (Time * Batch) 길이의 1D Tensor.
            # 따라서 states도 (Time * Batch, Dim)으로 맞춰야 함.
            pass

        # 에피소드 분리 로직은 단일 에이전트 기준이므로, 대량 병렬 학습시에는
        # 단순 미니배치 학습(Shuffle)으로 전환하는 것이 좋음.
        # RNN 사용 시에는 시퀀스 보존이 필요하지만, 여기서는 안정성을 위해 단순 PPO로 진행 권장.

        # 데이터셋 생성
        dataset_size = old_states.size(0)
        indices = torch.arange(dataset_size)

        avg_loss_all_epochs = 0
        avg_entropy_all_epochs = 0

        for _ in range(self.k_epochs):
            # Shuffle indices
            indices = indices[torch.randperm(dataset_size)]

            # Mini-batch loop (Batch size 예를 들어 64 or 256)
            batch_size = 256
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx]

                mb_states = old_states[batch_idx]  # (B, Dim)
                mb_actions = old_actions[batch_idx]  # (B, 2)
                mb_old_logprobs = old_logprobs[batch_idx]
                mb_rewards = rewards[batch_idx]

                # Forward (RNN hidden state is lost in simple batching - limitation)
                # RNN을 쓰더라도 여기서는 hidden=None으로 초기화하여 단기 기억만 사용하거나,
                # 전체 시퀀스를 유지해야 하는데 코드가 복잡해짐. 우선 실행을 위해 hidden=None 처리.
                # Model forward accepts (Batch, Dim) -> converts to (Batch, 1, Dim)
                logits_tuple, state_values, _ = self.policy(mb_states)

                target_logits, role_logits = logits_tuple

                # (Batch, 1, Dim) -> (Batch, Dim)
                if target_logits.dim() == 3:
                    target_logits = target_logits.squeeze(1)
                    role_logits = role_logits.squeeze(1)

                state_values = state_values.view(-1)

                dist_target = Categorical(logits=target_logits)
                dist_role = Categorical(logits=role_logits)

                logprobs_target = dist_target.log_prob(mb_actions[:, 0])
                logprobs_role = dist_role.log_prob(mb_actions[:, 1])
                logprobs = logprobs_target + logprobs_role

                dist_entropy = dist_target.entropy() + dist_role.entropy()

                ratios = torch.exp(logprobs - mb_old_logprobs)

                advantages = mb_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, mb_rewards)
                    - self.entropy_coef * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                avg_loss_all_epochs += loss.mean().item()
                avg_entropy_all_epochs += dist_entropy.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        # 총 배치가 몇 번 돌았는지 계산
        total_batches = self.k_epochs * (dataset_size // batch_size + 1)

        return {
            "loss": avg_loss_all_epochs / total_batches if total_batches > 0 else 0,
            "entropy": (
                avg_entropy_all_epochs / total_batches if total_batches > 0 else 0
            ),
        }

    def _split_episodes(self, data_list, is_terminals):
        # (Not used in batch update mode)
        pass
