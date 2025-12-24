from typing import List
from core.characters.rational import RationalCharacter
from core.characters.base import BaseCharacter


class CopyKitten(RationalCharacter):
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        # self.vote_history 사용
        repeat_offenders = [
            pid
            for pid, count in enumerate(self.vote_history)
            if count >= 2 and players[pid].alive
        ]

        if repeat_offenders:
            RETALIATION_LOGIT = 50
            for pid in repeat_offenders:
                self.suspicion[pid] += RETALIATION_LOGIT

        return super().decide_claim(players)

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return super().decide_claim(players)
