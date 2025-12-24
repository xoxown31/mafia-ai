from typing import List
from core.characters.rational import RationalCharacter
from core.characters.base import BaseCharacter


class CopyCat(RationalCharacter):
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        # self.voted_by_last_turn 사용
        alive_attackers = [pid for pid in self.voted_by_last_turn if players[pid].alive]

        alive_non_attackers = [
            p.id
            for p in players
            if p.alive and p.id != self.id and p.id not in self.voted_by_last_turn
        ]

        # [용서]
        if alive_non_attackers:
            FORGIVENESS_LOGIT = 10
            for pid in alive_non_attackers:
                self.suspicion_logits[pid] -= FORGIVENESS_LOGIT

        # [복수]
        if alive_attackers:
            RETALIATION_LOGIT = 40
            penalty = RETALIATION_LOGIT / len(alive_attackers)
            for pid in alive_attackers:
                self.suspicion_logits[pid] += penalty

        return super().decide_claim(players)

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return super().decide_claim(players)
