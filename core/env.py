import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
import config


class MafiaEnv(gym.Env):
    def __init__(self, log_file=None):
        self.game = MafiaGame(log_file=log_file)
        
        self.action_space = spaces.Discrete(config.TOTAL_ACTIONS)

        obs_dim = 168
        
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(config.TOTAL_ACTIONS,), dtype=np.int8
                ),
            }
        )
        
        self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        self.cumulative_accusation_count = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        self.cumulative_vote_count = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        self.cumulative_accusation_matrix = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        self.total_accusations = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        self.cumulative_accusation_count = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        self.cumulative_vote_count = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        self.cumulative_accusation_matrix = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
        self.total_accusations = 0
        return self._encode_observation(status), {}

    def step(self, action):
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        claim_role = -1
        target_id = -1
        accused_role = -1
        
        if action == config.ACTION_SILENT:
            pass
        elif config.ACTION_CLAIM_CITIZEN <= action <= config.ACTION_CLAIM_MAFIA:
            claim_role = action - config.ACTION_CLAIM_CITIZEN
        elif config.ACTION_ACCUSE_START <= action <= config.ACTION_ACCUSE_END:
            accuse_idx = action - config.ACTION_ACCUSE_START
            target_id = accuse_idx // 4
            accused_role = accuse_idx % 4
        
        status, done, win = self.game.process_turn(claim_role, target_id, accused_role)
        
        if prev_phase == config.PHASE_DAY_VOTE:
            self.last_vote_record = np.zeros((config.PLAYER_COUNT, config.PLAYER_COUNT), dtype=np.float32)
            for i, player in enumerate(self.game.players):
                if hasattr(player, 'voted_by_last_turn'):
                    for voter_id in player.voted_by_last_turn:
                        self.last_vote_record[voter_id][i] = 1.0
                        self.cumulative_vote_count[i] += 1.0
        
        if prev_phase == config.PHASE_DAY_DISCUSSION:
            for player in self.game.players:
                if hasattr(player, 'claimed_target') and player.claimed_target != -1:
                    if 0 <= player.claimed_target < config.PLAYER_COUNT:
                        self.cumulative_accusation_count[player.claimed_target] += 1.0
                        self.cumulative_accusation_matrix[player.id][player.claimed_target] += 1.0
                        self.total_accusations += 1

        reward = 0.0

        if done:
            if win:
                reward += 100.0
                reward += (config.MAX_DAYS - self.game.day_count) * 0.5
            else:
                reward -= 50.0

        if not self.game.players[my_id].alive:
            reward -= 1.0
        else:
            reward += 0.2

        if self.game.players[my_id].alive:
            phase = prev_phase  # 보상은 이전 페이즈 기준으로 계산
            
            # 역할 주장 보상
            if claim_role != -1:
                if claim_role == my_role:
                    # 진실 주장
                    reward += 1.0
                    if my_role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
                        reward += 2.0  # 특수 역할은 밝히기 위험도 높음
                else:
                    # 거짓 주장
                    if my_role == config.ROLE_MAFIA:
                        reward += 0.5  # 마피아는 위장 허용
                    else:
                        reward -= 1.0  # 시민 팀은 거짓 주장 페널티
            
            # 타겟 지목 및 역할 추정 보상
            if target_id != -1:
                if my_role == config.ROLE_CITIZEN:
                    reward += self._calculate_citizen_reward(target_id, accused_role, phase)
                elif my_role == config.ROLE_MAFIA:
                    reward += self._calculate_mafia_reward(target_id, accused_role, phase)
                elif my_role == config.ROLE_POLICE:
                    reward += self._calculate_citizen_reward(target_id, accused_role, phase)
                    reward += self._calculate_police_reward(target_id, phase)
                elif my_role == config.ROLE_DOCTOR:
                    reward += self._calculate_citizen_reward(target_id, accused_role, phase)
                    reward += self._calculate_doctor_reward(prev_alive, target_id, phase)

        return self._encode_observation(status), reward, done, False, {}

    def _get_action_mask(self):
        """
        액션 마스크 (37개):
        - 0: 침묵
        - 1~4: 역할 주장 (시민/경찰/의사/마피아)
        - 5~36: 타겟 지목 + 역할 추정 (8명 × 4역할 = 32개)
        """
        mask = np.zeros(config.TOTAL_ACTIONS, dtype=np.int8)
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        phase = self.game.phase

        # === 0: 침묵 (항상 허용) ===
        mask[config.ACTION_SILENT] = 1

        # === 1~4: 역할 주장 (토론 단계에만) ===
        if phase == config.PHASE_DAY_DISCUSSION:
            mask[config.ACTION_CLAIM_CITIZEN:config.ACTION_CLAIM_MAFIA + 1] = 1

        # === 5~36: 타겟 지목 + 역할 추정 ===
        for target_id in range(config.PLAYER_COUNT):
            # 기본 조건: 살아있는 플레이어만 지목 가능
            if not self.game.players[target_id].alive:
                # 죽은 플레이어는 모든 역할 추정 불가
                for role in range(4):
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                    mask[action_idx] = 0
                continue

            # 페이즈별 제약
            if phase == config.PHASE_DAY_DISCUSSION:
                # 토론: 자신 제외 모두 지목 가능
                if target_id == my_id:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0
                else:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 1

            elif phase == config.PHASE_DAY_VOTE:
                # 투표: 자신 제외 모두 투표 가능 (역할 추정 불필요, 첫 번째 역할만 사용)
                if target_id == my_id:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0
                else:
                    # 투표는 단순 지목이므로 첫 번째 슬롯만 허용
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                    mask[action_idx] = 1
                    # 나머지는 불허
                    for role in range(1, 4):
                        mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

            elif phase == config.PHASE_NIGHT:
                # 밤: 역할별 행동 제약
                if my_role == config.ROLE_MAFIA:
                    # 마피아: 동료 마피아 제외
                    if self.game.players[target_id].role == config.ROLE_MAFIA:
                        for role in range(4):
                            action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                            mask[action_idx] = 0
                    else:
                        # 첫 번째 슬롯만 허용 (밤에는 역할 추정 불필요)
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                        mask[action_idx] = 1
                        for role in range(1, 4):
                            mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

                elif my_role == config.ROLE_POLICE:
                    # 경찰: 자신 제외 모두 조사 가능
                    if target_id == my_id:
                        for role in range(4):
                            action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                            mask[action_idx] = 0
                    else:
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                        mask[action_idx] = 1
                        for role in range(1, 4):
                            mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

                elif my_role == config.ROLE_DOCTOR:
                    # 의사: 자신 포함 모두 치료 가능
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                    mask[action_idx] = 1
                    for role in range(1, 4):
                        mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

                elif my_role == config.ROLE_CITIZEN:
                    # 시민: 밤에는 행동 불가
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0

        return mask

    def _calculate_citizen_reward(self, target_id, accused_role, phase):
        """시민 팀 보상: target_id 지목, accused_role 추정"""
        if target_id == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]

            if phase == config.PHASE_DAY_VOTE:
                # 투표 단계: 마피아 투표 시 높은 보상
                if target.role == config.ROLE_MAFIA:
                    reward += 5.0
                    if not target.alive:
                        reward += 3.0
                elif target.role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
                    reward -= 4.0
                elif target.role == config.ROLE_CITIZEN:
                    reward -= 1.0
            
            elif phase == config.PHASE_DAY_DISCUSSION:
                # 토론 단계: 정확한 역할 추정 시 소폭 보상
                if accused_role != -1:
                    if accused_role == config.ROLE_MAFIA and target.role == config.ROLE_MAFIA:
                        reward += 1.5  # 마피아 정확히 지목
                    elif accused_role != config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward += 0.5  # 시민 팀 정확히 추정
                    elif accused_role == config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward -= 0.5  # 시민을 마피아로 오인
                
        return reward

    def _calculate_mafia_reward(self, target_id, accused_role, phase):
        """마피아 보상: 시민 팀 제거 및 잘못된 지목 유도"""
        if target_id == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]

            if phase == config.PHASE_DAY_VOTE:
                # 투표 단계: 시민 팀 제거 시 보상
                if target.role == config.ROLE_POLICE:
                    reward += 7.0
                    if not target.alive:
                        reward += 3.0
                elif target.role == config.ROLE_DOCTOR:
                    reward += 5.0
                    if not target.alive:
                        reward += 2.0
                elif target.role == config.ROLE_CITIZEN:
                    reward += 2.0
                    if not target.alive:
                        reward += 1.0
                elif target.role == config.ROLE_MAFIA:
                    reward -= 15.0  # 동료 마피아 투표 시 큰 페널티

            elif phase == config.PHASE_NIGHT:
                # 밤 단계: 고가치 타겟 제거
                if target.role == config.ROLE_POLICE:
                    reward += 8.0
                    if not target.alive:
                        reward += 2.0
                elif target.role == config.ROLE_DOCTOR:
                    reward += 6.0
                    if not target.alive:
                        reward += 1.5
                elif target.role == config.ROLE_CITIZEN:
                    reward += 3.0
                    if not target.alive:
                        reward += 1.0
            
            elif phase == config.PHASE_DAY_DISCUSSION:
                # 토론 단계: 잘못된 역할 추정 유도 (위장)
                if accused_role != -1:
                    if accused_role != config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward += 0.5  # 시민을 다른 역할로 지목 (혼란 유도)
                
        return reward

    def _calculate_police_reward(self, target_id, phase):
        """경찰 보상: 마피아 조사"""
        if target_id == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]
            if target.role == config.ROLE_MAFIA:
                reward += 7.0  # 마피아 발견
            else:
                reward += 0.5  # 시민 확인
        return reward

    def _calculate_doctor_reward(self, prev_alive, target_id, phase):
        """의사 보상: 치료 성공"""
        if target_id == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        current_alive_count = sum(self.game.alive_status)
        prev_alive_count = sum(prev_alive)
        
        if current_alive_count == prev_alive_count:
            reward += 8.0  # 치료 성공 (사망자 없음)
            
            if 0 <= target_id < len(self.game.players):
                target = self.game.players[target_id]
                if target.role == config.ROLE_POLICE:
                    reward += 2.0  # 경찰 보호
                elif target.role == config.ROLE_DOCTOR:
                    reward += 1.5  # 자신 보호
                elif target.role == config.ROLE_CITIZEN:
                    reward += 1.0  # 시민 보호
        else:
            reward += 0.3  # 치료 실패해도 소폭 보상

        return reward

    def _encode_observation(self, status):
        alive_vector = np.array(status["alive_status"], dtype=np.float32)
        
        my_role_id = status["roles"][status["id"]]
        role_one_hot = np.zeros(4, dtype=np.float32)
        role_one_hot[my_role_id] = 1.0
        
        claim_status = np.zeros(config.PLAYER_COUNT, dtype=np.float32)
        for player in self.game.players:
            if hasattr(player, 'claimed_role') and player.claimed_role != -1:
                claim_status[player.id] = player.claimed_role / 3.0
        
        if self.total_accusations > 0:
            accusation_matrix = self.cumulative_accusation_matrix / self.total_accusations
        else:
            accusation_matrix = self.cumulative_accusation_matrix
        accusation_flat = accusation_matrix.flatten()
        
        last_vote_flat = self.last_vote_record.flatten()
        cum_accusation_normalized = np.clip(self.cumulative_accusation_count / 10.0, 0.0, 1.0)
        cum_vote_normalized = np.clip(self.cumulative_vote_count / 10.0, 0.0, 1.0)
        day_normalized = np.array([min(self.game.day_count / config.MAX_DAYS, 1.0)], dtype=np.float32)
        
        phase_map = {
            config.PHASE_DAY_DISCUSSION: 0,
            config.PHASE_DAY_VOTE: 1,
            config.PHASE_NIGHT: 2,
        }
        phase_idx = phase_map.get(self.game.phase, 0)
        phase_onehot = np.zeros(3, dtype=np.float32)
        phase_onehot[phase_idx] = 1.0
        
        observation = np.concatenate([
            alive_vector,
            role_one_hot,
            claim_status,
            accusation_flat,
            last_vote_flat,
            cum_accusation_normalized,
            cum_vote_normalized,
            day_normalized,
            phase_onehot
        ])
        
        action_mask = self._get_action_mask()
        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        phase_str = (
            ["Claim", "Discussion", "Vote", "Night"][self.game.phase]
            if isinstance(self.game.phase, int)
            else self.game.phase
        )
        alive_indices = [i for i, alive in enumerate(self.game.alive_status) if alive]
        print(f"[Day {self.game.day_count}] {phase_str} | Alive: {alive_indices}")