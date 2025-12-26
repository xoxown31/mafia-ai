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
        reveal_role, target_id, accused_role = -1, -1, -1
        
        if action == config.ACTION_SILENT:
            pass
        elif config.ACTION_REVEAL_CITIZEN <= action <= config.ACTION_REVEAL_MAFIA:
            reveal_role = action - config.ACTION_REVEAL_CITIZEN
        elif config.ACTION_ACCUSE_START <= action <= config.ACTION_ACCUSE_END:
            accuse_idx = action - config.ACTION_ACCUSE_START
            target_id = accuse_idx // 4
            accused_role = accuse_idx % 4
        
        status, done, win = self.game.process_turn(reveal_role, target_id, accused_role)
        
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
            phase = prev_phase
            
            if reveal_role != -1:
                if reveal_role == my_role:
                    reward += 1.0
                    if my_role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
                        reward += 2.0
                else:
                    if my_role == config.ROLE_MAFIA:
                        reward += 0.5
                    else:
                        reward -= 1.0
            
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
        mask = np.zeros(config.TOTAL_ACTIONS, dtype=np.int8)
        my_id = self.game.players[0].id
        my_role = self.game.players[my_id].role
        phase = self.game.phase

        mask[config.ACTION_SILENT] = 1

        if phase == config.PHASE_DAY_DISCUSSION:
            mask[config.ACTION_REVEAL_CITIZEN:config.ACTION_REVEAL_MAFIA + 1] = 1

        for target_id in range(config.PLAYER_COUNT):
            if not self.game.players[target_id].alive:
                for role in range(4):
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                    mask[action_idx] = 0
                continue

            if phase == config.PHASE_DAY_DISCUSSION:
                if target_id == my_id:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0
                else:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 1

            elif phase == config.PHASE_DAY_VOTE:
                if target_id == my_id:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0
                else:
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                    mask[action_idx] = 1
                    for role in range(1, 4):
                        mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

            elif phase == config.PHASE_NIGHT:
                if my_role == config.ROLE_MAFIA:
                    if self.game.players[target_id].role == config.ROLE_MAFIA:
                        for role in range(4):
                            action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                            mask[action_idx] = 0
                    else:
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                        mask[action_idx] = 1
                        for role in range(1, 4):
                            mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

                elif my_role == config.ROLE_POLICE:
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
                    action_idx = config.ACTION_ACCUSE_START + (target_id * 4)
                    mask[action_idx] = 1
                    for role in range(1, 4):
                        mask[config.ACTION_ACCUSE_START + (target_id * 4 + role)] = 0

                elif my_role == config.ROLE_CITIZEN:
                    for role in range(4):
                        action_idx = config.ACTION_ACCUSE_START + (target_id * 4 + role)
                        mask[action_idx] = 0

        return mask

    def _calculate_citizen_reward(self, target_id, accused_role, phase):
        if target_id == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]

            if phase == config.PHASE_DAY_VOTE:
                if target.role == config.ROLE_MAFIA:
                    reward += 5.0
                    if not target.alive:
                        reward += 3.0
                elif target.role in [config.ROLE_POLICE, config.ROLE_DOCTOR]:
                    reward -= 4.0
                elif target.role == config.ROLE_CITIZEN:
                    reward -= 1.0
            
            elif phase == config.PHASE_DAY_DISCUSSION:
                if accused_role != -1:
                    if accused_role == config.ROLE_MAFIA and target.role == config.ROLE_MAFIA:
                        reward += 1.5
                    elif accused_role != config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward += 0.5
                    elif accused_role == config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward -= 0.5
                
        return reward

    def _calculate_mafia_reward(self, target_id, accused_role, phase):
        if target_id == -1:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]

            if phase == config.PHASE_DAY_VOTE:
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
                    reward -= 15.0

            elif phase == config.PHASE_NIGHT:
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
                if accused_role != -1:
                    if accused_role != config.ROLE_MAFIA and target.role != config.ROLE_MAFIA:
                        reward += 0.5
                
        return reward

    def _calculate_police_reward(self, target_id, phase):
        if target_id == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        if 0 <= target_id < len(self.game.players):
            target = self.game.players[target_id]
            if target.role == config.ROLE_MAFIA:
                reward += 7.0
            else:
                reward += 0.5
        return reward

    def _calculate_doctor_reward(self, prev_alive, target_id, phase):
        if target_id == -1 or phase != config.PHASE_NIGHT:
            return 0.0
            
        reward = 0.0
        
        current_alive_count = sum(self.game.alive_status)
        prev_alive_count = sum(prev_alive)
        
        if current_alive_count == prev_alive_count:
            reward += 8.0
            
            if 0 <= target_id < len(self.game.players):
                target = self.game.players[target_id]
                if target.role == config.ROLE_POLICE:
                    reward += 2.0
                elif target.role == config.ROLE_DOCTOR:
                    reward += 1.5
                elif target.role == config.ROLE_CITIZEN:
                    reward += 1.0
        else:
            reward += 0.3

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