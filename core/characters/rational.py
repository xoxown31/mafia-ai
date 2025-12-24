import random
from typing import List
import config
from core.characters.base import BaseCharacter


class RationalCharacter(BaseCharacter):
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        # self.id 사용
        alive_indices = self._get_alive_ids(players, exclude_me=True)
        if not alive_indices:
            return -1

        # self.suspicion 사용
        max_susp = max(self.suspicion[i] for i in alive_indices)
        candidates = [i for i in alive_indices if self.suspicion[i] == max_susp]
        return random.choice(candidates)

    def update_suspicion(self, speaker: "BaseCharacter", target_id: int):
        # self.suspicion / self.suspicion_logits 사용
        if self.suspicion[speaker.id] < 40:
            self.suspicion_logits[target_id] += 50

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return self.decide_claim(players)

    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]
            if not targets:
                return alive_ids[0]
            return max(targets, key=lambda i: self.suspicion[i])

        elif current_role == config.ROLE_DOCTOR:
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: self.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            return max(alive_ids, key=lambda i: self.suspicion[i])

        return -1
