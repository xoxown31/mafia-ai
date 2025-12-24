from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class Maverick(BaseCharacter):
    """
    독불장군 (Maverick)
    - 청개구리 스타일
    - 남들이 의심하는 사람은 오히려 믿음
    - 남들이 믿는 사람은 의심
    - 반대 의견을 선호하는 독립적인 사고
    """

    def update_belief(self, game_status: dict):
        """
        다수의 의견과 반대로 행동:
        - 많은 표를 받은 사람의 마피아 점수 감소
        - 적은 표를 받거나 무시당한 사람 의심
        """
        if 'vote_results' in game_status:
            vote_counts = {}
            all_voters = set()
            
            for voter_id, target_id in game_status['vote_results']:
                if target_id >= 0:
                    vote_counts[target_id] = vote_counts.get(target_id, 0) + 1
                    all_voters.add(voter_id)
            
            # 가장 많은 표를 받은 사람
            if vote_counts:
                max_votes = max(vote_counts.values())
                
                for target_id, count in vote_counts.items():
                    if count == max_votes:
                        # 다수가 의심하는 사람 = 오히려 믿음 (청개구리)
                        self.belief[target_id, 3] -= 80  # 마피아 의심 대폭 감소
                        self.belief[target_id, 0] += 50  # 시민 신뢰 증가
                    elif count <= 1:
                        # 소수만 의심하는 사람 = 오히려 의심
                        self.belief[target_id, 3] += 60
            
            # 아무도 투표하지 않은 사람들 (조용히 있는 사람들)을 의심
            for pid in range(config.PLAYER_COUNT):
                if pid != self.id and pid not in vote_counts and pid in all_voters:
                    # 투표는 했지만 표를 받지 않은 = 눈에 띄지 않는 = 의심스러움
                    self.belief[pid, 3] += 40
        
        # Claim 정보 역분석
        if 'claims' in game_status:
            claim_counts = {}
            for claimer_id, target_id in game_status['claims']:
                if target_id >= 0:
                    claim_counts[target_id] = claim_counts.get(target_id, 0) + 1
            
            # 많은 사람이 지목한 대상의 점수 감소 (청개구리)
            for target_id, count in claim_counts.items():
                if count >= 2:
                    self.belief[target_id, 3] -= count * 30
                    self.belief[target_id, 0] += count * 20
        
        # 자신의 belief는 확실하게
        self.belief[self.id, self.role] = 100.0
