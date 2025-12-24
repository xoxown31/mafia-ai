from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class CopyKitten(BaseCharacter):
    """
    너그러운 따라쟁이
    - 2회 이상 나를 공격한 사람만 의심
    - 한 번의 실수는 용서
    """

    def update_belief(self, game_status: dict):
        """
        반복 공격자만 의심하는 로직
        """
        # 누적 투표 히스토리 확인
        repeat_offenders = [
            pid for pid, count in enumerate(self.vote_history)
            if count >= 2 and pid != self.id
        ]
        
        if repeat_offenders:
            RETALIATION_SCORE = 120.0  # 2회 이상 공격시 강력한 보복
            for pid in repeat_offenders:
                # 반복 공격자는 마피아로 의심
                self.belief[pid, 3] += RETALIATION_SCORE
                self.belief[pid, 0] -= 60
        
        # 한 번만 공격한 사람은 용서 (점수 약간 회복)
        single_voters = [
            pid for pid, count in enumerate(self.vote_history)
            if count == 1 and pid != self.id
        ]
        
        FORGIVENESS_SCORE = 30
        for pid in single_voters:
            self.belief[pid, 0] += FORGIVENESS_SCORE
            self.belief[pid, 3] -= FORGIVENESS_SCORE / 2
        
        # 자신의 belief 확실하게
        self.belief[self.id, self.role] = 100.0
