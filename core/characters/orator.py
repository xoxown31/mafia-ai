from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class Orator(BaseCharacter):
    """
    웅변가 (Orator)
    - 자기 직관 가중치 극대화 (5.0 이상)
    - 타인의 의견을 거의 무시
    - 독립적인 판단을 중시
    """
    
    INTUITION_WEIGHT = 5.0  # 자기 직관 가중치

    def update_belief(self, game_status: dict):
        """
        자기 직관만 믿고, 타인 의견 무시
        게임 이벤트(투표, 발언 등)에서 자신만의 논리로 belief 업데이트
        """
        # 예시: 최근 투표 결과에서 소수 의견을 낸 사람들의 마피아 점수 증가
        if 'vote_results' in game_status:
            vote_counts = {}
            for voter_id, target_id in game_status['vote_results']:
                if target_id not in vote_counts:
                    vote_counts[target_id] = []
                vote_counts[target_id].append(voter_id)
            
            # 가장 많이 받은 표의 투표자들 = 다수파
            # 소수파를 의심
            if vote_counts:
                majority_target = max(vote_counts.keys(), key=lambda t: len(vote_counts[t]))
                for target, voters in vote_counts.items():
                    if target != majority_target:
                        # 소수 의견을 낸 사람들을 의심 (자기 직관 강화)
                        for voter_id in voters:
                            self.belief[voter_id, 3] += 15 * self.INTUITION_WEIGHT  # 마피아 의심 대폭 증가
        
        # 자신의 신념을 강화
        self.belief[self.id, self.role] = 100.0
