from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class CopyCat(BaseCharacter):
    """
    복수하되 용서도 하는 캐릭터
    - 나를 공격한 사람은 의심
    - 나를 공격하지 않은 사람은 신뢰
    """

    def update_belief(self, game_status: dict):
        """
        공격자와 비공격자를 구분하여 belief 업데이트
        """
        # 지난 턴에 나를 투표한 사람들
        if self.voted_by_last_turn:
            RETALIATION_SCORE = 100.0  # Grudger와 동등한 보복심
            penalty_per_attacker = RETALIATION_SCORE / len(self.voted_by_last_turn)
            
            for attacker_id in self.voted_by_last_turn:
                # 공격자는 마피아 의심 증가
                self.belief[attacker_id, 3] += penalty_per_attacker
                self.belief[attacker_id, 0] -= penalty_per_attacker / 2
        
        # 나를 공격하지 않은 살아있는 사람들 (용서)
        if 'alive_players' in game_status:
            alive_non_attackers = [
                pid for pid in game_status['alive_players']
                if pid != self.id and pid not in self.voted_by_last_turn
            ]
            
            FORGIVENESS_SCORE = 40
            for pid in alive_non_attackers:
                # 비공격자는 시민으로 신뢰
                self.belief[pid, 0] += FORGIVENESS_SCORE
                self.belief[pid, 3] -= FORGIVENESS_SCORE / 2
        
        # 자신의 belief 확실하게
        self.belief[self.id, self.role] = 100.0
