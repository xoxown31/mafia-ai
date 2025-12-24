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
            FORGIVENESS_SCORE = 10
            for pid in alive_non_attackers:
                # suspicion_logits -> suspicion
                self.suspicion[pid] -= FORGIVENESS_SCORE 

        # [복수]
        if alive_attackers:
            RETALIATION_SCORE = 40
            penalty = RETALIATION_SCORE / len(alive_attackers)
            for pid in alive_attackers:
                # suspicion_logits -> suspicion
                self.suspicion[pid] += penalty

        return super().decide_claim(players)
    
    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return super().decide_claim(players)
