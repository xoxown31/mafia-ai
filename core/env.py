import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.game import MafiaGame
from config import *


class MafiaEnv(gym.Env):
    def __init__(self, log_file=None):
        super(MafiaEnv, self).__init__()
        self.game = MafiaGame(log_file=log_file)
        
        # === [Action Space 확장 v2.0] ===
        # 기존: 0~7번 플레이어 지목 + 8번 기권 (총 9개)
        # 확장: + 9~12번 역할 주장 (시민/경찰/의사/마피아) + 13~20번 역할 주장+지목 복합 액션
        # 총 21개 액션:
        # - 0~7: 단순 지목 (토론/투표/밤)
        # - 8: 기권 (NO_ACTION)
        # - 9: 시민 주장 (CLAIM_CITIZEN)
        # - 10: 경찰 주장 (CLAIM_POLICE) 
        # - 11: 의사 주장 (CLAIM_DOCTOR)
        # - 12: 마피아 주장 (CLAIM_MAFIA - 학습 가능하도록)
        # - 13~20: 경찰 주장 + 0~7번 지목 (조사 결과 발표)
        self.action_space = spaces.Discrete(21)

        # Observation Space: 공적 정보만 포함 (기존 유지)
        # - alive_status: 8 (생존 여부)
        # - my_role: 4 (내 역할 one-hot)
        # - claim_status: 8 (각 플레이어가 주장한 역할: 0~3)
        # - accusation_matrix: 8*8=64 (누가 누구를 의심했는지)
        # - last_vote_matrix: 8*8=64 (직전 투표에서 누가 누구에게 투표했는지)
        # - day_count: 1 (현재 날짜, 정규화)
        # - phase_onehot: 3 (현재 페이즈: discussion, vote, night)
        # Total: 8 + 4 + 8 + 64 + 64 + 1 + 3 = 152
        obs_dim = 152
        
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(21,), dtype=np.int8  # 21개 액션으로 확장
                ),
            }
        )
        
        # 이전 턴의 투표 기록 저장
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        status = self.game.reset()
        # 투표 기록 초기화
        self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
        return self._encode_observation(status), {}

    def step(self, action):
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        
        # 턴 진행 전 상태 저장 (의사 보상 계산용)
        prev_alive = [p.alive for p in self.game.players]
        prev_phase = self.game.phase

        # === [확장된 액션 처리] ===
        ai_claim_role = -1  # AI가 주장할 역할 (-1: 주장 없음)
        processed_action = -1  # 실제 게임에 전달될 지목 액션
        
        if action == 8:
            # 기권
            processed_action = -1
        elif 0 <= action <= 7:
            # 단순 지목
            processed_action = action
        elif action == 9:
            # 시민 주장
            ai_claim_role = Role.CITIZEN
            processed_action = -1
        elif action == 10:
            # 경찰 주장
            ai_claim_role = Role.POLICE
            processed_action = -1
        elif action == 11:
            # 의사 주장
            ai_claim_role = Role.DOCTOR
            processed_action = -1
        elif action == 12:
            # 마피아 주장 (블러핑 전략 학습 가능)
            ai_claim_role = Role.MAFIA
            processed_action = -1
        elif 13 <= action <= 20:
            # 경찰 주장 + 지목 복합 액션
            ai_claim_role = Role.POLICE
            processed_action = action - 13  # 13 → 0, 14 → 1, ..., 20 → 7

        # 게임 진행 (역할 주장 정보도 전달)
        status, done, win = self.game.process_turn(processed_action, ai_claim_role)
        
        # === 투표 기록 업데이트 (PHASE_DAY_VOTE 종료 후) ===
        if prev_phase == Phase.DAY_VOTE:
            # 투표 매트릭스 업데이트
            self.last_vote_record = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
            for i, player in enumerate(self.game.players):
                if hasattr(player, 'voted_by_last_turn'):
                    for voter_id in player.voted_by_last_turn:
                        self.last_vote_record[voter_id][i] = 1.0

        # === [보상 함수 고도화 v2.0: Dense Reward] ===
        reward = 0.0

        # 1. 승패 보상 - 비중 대폭 감소 (초기 학습 촉진)
        if done:
            if win:
                reward += 30.0  # 승리 보상 (기존 100 → 30으로 감소)
                # 빨리 이길수록 추가 보상
                reward += (config.game.MAX_DAYS - self.game.day_count) * 1.0
            else:
                reward -= 15.0  # 패배 페널티 (기존 -50 → -15로 감소)

        # 2. 생존 보상 - 매 턴 피드백 제공
        if not self.game.players[my_id].alive:
            reward -= 2.0  # 죽음 페널티
        else:
            reward += 0.5  # 매 턴 생존 시 소량 보상 (학습 신호)

        # 3. 역할 기반 행동 보상 (명확한 보상만)
        if self.game.players[my_id].alive:
            phase = self.game.phase
            
            # === [역할 주장 보상] ===
            if ai_claim_role != -1:
                # 자신의 실제 역할을 주장하면 보상
                if ai_claim_role == my_role:
                    reward += 2.0  # 정직한 주장 보상
                    # 경찰/의사가 자신을 밝히면 추가 보상 (전략적 선택)
                    if my_role in [Role.POLICE, Role.DOCTOR]:
                        reward += 3.0
                else:
                    # 거짓 주장 (블러핑) - 마피아에게 유용할 수 있음
                    if my_role == Role.MAFIA:
                        reward += 1.0  # 마피아의 블러핑 허용
                    else:
                        reward -= 2.0  # 시민 팀의 거짓 주장은 페널티
            
            # === [행동 보상] ===
            if processed_action != -1:
                phase = self.game.phase
                
                if my_role == Role.CITIZEN:
                    reward += self._calculate_citizen_reward(processed_action, phase)
                elif my_role == Role.MAFIA:
                    reward += self._calculate_mafia_reward(processed_action, phase)
                elif my_role == Role.POLICE:
                    reward += self._calculate_citizen_reward(processed_action, phase)
                    reward += self._calculate_police_reward(processed_action, phase)
                elif my_role == Role.DOCTOR:
                    reward += self._calculate_citizen_reward(processed_action, phase)
                    reward += self._calculate_doctor_reward(prev_alive, processed_action, phase)

        return self._encode_observation(status), reward, done, False, {}

    def _get_action_mask(self):
        mask = np.ones(21, dtype=np.int8)  # 21개 액션으로 확장
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        phase = self.game.phase

        # === 0~7: 지목 액션 마스크 ===
        for i in range(config.game.PLAYER_COUNT):
            # 1. 이미 죽은 플레이어는 지목 불가
            if not self.game.players[i].alive:
                mask[i] = 0
                continue

            # 2. 낮 행동 제약 (자신 지목 불가)
            if phase == Phase.DAY_DISCUSSION or phase == Phase.DAY_VOTE:
                if i == my_id:
                    mask[i] = 0

            # 3. 밤 행동 제약
            elif phase == Phase.NIGHT:
                # 마피아: 동료 마피아 지목 불가
                if my_role == Role.MAFIA:
                    if self.game.players[i].role == Role.MAFIA:
                        mask[i] = 0
                # 경찰: 자신 조사 불가
                elif my_role == Role.POLICE:
                    if i == my_id:
                        mask[i] = 0
                # 의사: 자신 치료 가능 (제약 없음)

        # === 8: 기권 액션 마스크 ===
        if phase == Phase.DAY_DISCUSSION or phase == Phase.DAY_VOTE:
            mask[8] = 1  # 기권 허용
        else:
            mask[8] = 0  # 밤에는 기권 불가
        
        # === 9~12: 역할 주장 액션 마스크 (토론 단계에만 가능) ===
        if phase == Phase.DAY_DISCUSSION:
            mask[9:13] = 1  # 시민/경찰/의사/마피아 주장 모두 허용
        else:
            mask[9:13] = 0  # 토론 단계 외에는 주장 불가
        
        # === 13~20: 경찰 주장+지목 복합 액션 (토론 단계에만) ===
        if phase == Phase.DAY_DISCUSSION:
            for i in range(8):
                if self.game.players[i].alive and i != my_id:
                    mask[13 + i] = 1  # 살아있고 자신이 아닌 플레이어 지목 가능
                else:
                    mask[13 + i] = 0
        else:
            mask[13:21] = 0  # 토론 단계 외에는 불가

        return mask

    def _calculate_citizen_reward(self, action, phase):
        """시민 팀 공통 보상 로직 - Dense Reward 강화"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        # IndexError 방지
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 마피아를 지목하면 강력한 중간 보상
            if phase == Phase.DAY_VOTE:
                if target.role == Role.MAFIA:
                    reward += 15.0  # 마피아 투표 성공 (대폭 강화: 5 → 15)
                    # 투표 성공 시 추가 보상 (즉각적 피드백)
                    if not target.alive:  # 실제로 처형되었다면
                        reward += 10.0  # 처형 성공 추가 보상
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 8.0  # 중요 역할 지목 페널티 강화
                elif target.role == Role.CITIZEN:
                    reward -= 2.0  # 시민 지목 페널티
            
            # 낮 토론: 마피아를 의심하면 중간 보상 (학습 신호)
            elif phase == Phase.DAY_DISCUSSION:
                if target.role == Role.MAFIA:
                    reward += 3.0  # 마피아 의심 보상 강화 (1 → 3)
                elif target.role in [Role.POLICE, Role.DOCTOR]:
                    reward -= 1.0  # 중요 역할 의심 시 소량 페널티
                
        return reward

    def _calculate_mafia_reward(self, action, phase):
        """마피아 보상 로직 - Dense Reward 대폭 강화"""
        if action == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]

            # 낮 투표: 시민 팀 제거 성공 (중간 보상 강화)
            if phase == Phase.DAY_VOTE:
                if target.role == Role.POLICE:
                    reward += 20.0  # 경찰 제거 최우선 (대폭 강화: 8 → 20)
                    if not target.alive:  # 실제 처형 성공
                        reward += 15.0  # 처형 성공 추가 보상
                elif target.role == Role.DOCTOR:
                    reward += 15.0  # 의사 제거 (5 → 15)
                    if not target.alive:
                        reward += 10.0
                elif target.role == Role.CITIZEN:
                    reward += 5.0  # 시민 제거 (2 → 5)
                    if not target.alive:
                        reward += 3.0
                elif target.role == Role.MAFIA:
                    reward -= 25.0  # 동료 마피아 지목 심각한 페널티 (강화)

            # 밤 행동: 중요 역할 제거 (매우 강력한 중간 보상)
            elif phase == Phase.NIGHT:
                if target.role == Role.POLICE:
                    reward += 25.0  # 경찰 제거 최우선 (대폭 강화: 10 → 25)
                    # 실제 킬 성공 확인 (의사 치료 실패)
                    if not target.alive:
                        reward += 15.0  # 킬 성공 추가 보상
                elif target.role == Role.DOCTOR:
                    reward += 18.0  # 의사 제거 (7 → 18)
                    if not target.alive:
                        reward += 12.0
                elif target.role == Role.CITIZEN:
                    reward += 8.0  # 시민 제거 (3 → 8)
                    if not target.alive:
                        reward += 5.0
                
        return reward

    def _calculate_police_reward(self, action, phase):
        """경찰 특수 보상 - 밤에 조사 성공 (Dense Reward 대폭 강화)"""
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
            
        reward = 0.0
        
        if 0 <= action < len(self.game.players):
            target = self.game.players[action]
            if target.role == Role.MAFIA:
                reward += 20.0  # 마피아 발견 성공 (대폭 강화: 8 → 20)
                # 조사 성공은 매우 중요한 정보 획득
            else:
                reward += 2.0  # 조사 자체에 대한 보상 증가 (0.5 → 2.0)
        return reward

    def _calculate_doctor_reward(self, prev_alive, action, phase):
        """의사 특수 보상 - 치료 성공 (Dense Reward 대폭 강화)"""
        if action == -1 or phase != Phase.NIGHT:
            return 0.0
            
        reward = 0.0
        
        # 치료 성공 확인 (사망자가 없었으면 치료 성공)
        current_alive_count = sum(self.game.alive_status)
        prev_alive_count = sum(prev_alive)
        
        if current_alive_count == prev_alive_count:
            # 치료 성공 (매우 중요한 행동)
            reward += 25.0  # 치료 성공 보상 대폭 강화 (10 → 25)
            
            # 중요 역할을 살렸으면 추가 보상
            if 0 <= action < len(self.game.players):
                target = self.game.players[action]
                if target.role == Role.POLICE:
                    reward += 15.0  # 경찰 구출 (5 → 15)
                elif target.role == Role.DOCTOR:
                    reward += 10.0  # 자기 자신 또는 다른 의사
                elif target.role == Role.CITIZEN:
                    reward += 5.0  # 시민 구출
        else:
            # 치료 실패 시에도 시도에 대한 소량 보상
            reward += 1.0  # 행동 자체에 대한 피드백

        return reward

    def _encode_observation(self, status):
        """
        공개 정보만을 사용한 관측 인코딩 (RationalCharacter의 belief 사용 금지)
        
        구성:
        1. alive_status (8): 각 플레이어 생존 여부
        2. my_role (4): 내 역할 one-hot (citizen, police, doctor, mafia)
        3. claim_status (8): 각 플레이어가 주장한 역할 (0~3, 정규화)
        4. accusation_matrix (64): 누가 누구를 의심했는지 (8x8 평탄화)
        5. last_vote_matrix (64): 직전 투표 기록 (8x8 평탄화)
        6. day_count (1): 현재 날짜 (정규화)
        7. phase_onehot (3): 현재 페이즈 (discussion, vote, night)
        
        Total: 152차원
        """
        # 1. 생존 상태 (8)
        alive_vector = np.array(status["alive_status"], dtype=np.float32)
        
        # 2. 내 역할 one-hot (4)
        my_role_id = status["roles"][status["id"]]
        role_one_hot = np.zeros(4, dtype=np.float32)
        role_one_hot[my_role_id] = 1.0
        
        # 3. Claim Status (8): 각 플레이어가 주장한 역할
        claim_status = np.zeros(config.game.PLAYER_COUNT, dtype=np.float32)
        for player in self.game.players:
            # claimed_role 속성이 있으면 사용, 없으면 0 (주장 없음)
            if hasattr(player, 'claimed_role') and player.claimed_role != -1:
                claim_status[player.id] = player.claimed_role / 3.0  # 0~3을 0~1로 정규화
        
        # 4. Accusation Matrix (64): 현재 턴에서 누가 누구를 지목했는지
        accusation_matrix = np.zeros((config.game.PLAYER_COUNT, config.game.PLAYER_COUNT), dtype=np.float32)
        for player in self.game.players:
            if hasattr(player, 'claimed_target') and player.claimed_target != -1:
                # player.id가 claimed_target을 지목했음
                if 0 <= player.claimed_target < config.game.PLAYER_COUNT:
                    accusation_matrix[player.id][player.claimed_target] = 1.0
        accusation_flat = accusation_matrix.flatten()
        
        # 5. Last Vote Matrix (64): 직전 투표 기록 (이미 self.last_vote_record에 저장됨)
        last_vote_flat = self.last_vote_record.flatten()
        
        # 6. Day Count (1): 정규화 (0~1 범위, MAX_DAYS 기준)
        day_normalized = np.array([min(self.game.day_count / config.game.MAX_DAYS, 1.0)], dtype=np.float32)
        
        # 7. Phase One-hot (3)
        phase_map = {
            Phase.DAY_DISCUSSION: 0,
            Phase.DAY_VOTE: 1,
            Phase.NIGHT: 2,
        }
        phase_idx = phase_map.get(self.game.phase, 0)
        phase_onehot = np.zeros(3, dtype=np.float32)
        phase_onehot[phase_idx] = 1.0
        
        # 전체 observation 결합
        observation = np.concatenate([
            alive_vector,      # 8
            role_one_hot,      # 4
            claim_status,      # 8
            accusation_flat,   # 64
            last_vote_flat,    # 64
            day_normalized,    # 1
            phase_onehot       # 3
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