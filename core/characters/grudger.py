from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class Grudger(BaseCharacter):
    """
    원한주의자 (Grudger)
    - 지난 턴에 나를 찍은 사람을 절대 용서하지 않음
    - 복수심이 강하고 감정적인 판단
    - 한 번 적으로 인식하면 계속 의심
    """

    def update_belief(self, game_status: dict):
        """
        나를 투표한 사람들의 마피아 점수를 대폭 증가
        """
        # 지난 턴에 나를 투표한 사람들 처리
        if self.voted_by_last_turn:
            RETALIATION_SCORE = 100.0  # 10배 증가: 무조건 보복
            for attacker_id in self.voted_by_last_turn:
                # 마피아 의심 대폭 증가
                self.belief[attacker_id, 3] += RETALIATION_SCORE
                # 시민 점수는 감소
                self.belief[attacker_id, 0] -= 50
        
        # 누적된 투표 히스토리도 반영 (나를 여러 번 찍은 사람)
        for pid, count in enumerate(self.vote_history):
            if count >= 2 and pid != self.id:
                # 반복 공격자는 더욱 의심
                self.belief[pid, 3] += count * 50
        
        # 자신의 belief는 확실하게 유지
        self.belief[self.id, self.role] = 100.0
