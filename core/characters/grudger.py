from typing import List
from core.characters.rational import RationalCharacter
from core.characters.base import BaseCharacter


class Grudger(RationalCharacter):
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        alive_attackers = [pid for pid in self.voted_by_last_turn if players[pid].alive]

        if alive_attackers:
            RETALIATION_LOGIT = 100
            for pid in alive_attackers:
                self.suspicion_logits[pid] += RETALIATION_LOGIT

        return super().decide_claim(players)

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return super().decide_claim(players)
