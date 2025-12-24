from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class Follower(BaseCharacter):
    """
    소시민 (Follower)
    - 타인의 투표 현황(다수결)을 크게 반영
    - 자기 판단보다는 집단의 의견을 따름
    - 안전한 선택을 선호
    """

    def update_belief(self, game_status: dict):
        """
        다수결 로직: 많은 사람들이 의심하는 사람을 따라서 의심
        """
        # 투표 결과 분석
        if 'vote_results' in game_status:
            vote_counts = {}
            for voter_id, target_id in game_status['vote_results']:
                if target_id >= 0:  # 유효한 투표
                    vote_counts[target_id] = vote_counts.get(target_id, 0) + 1
            
            # 투표 1위 찾기
            if vote_counts:
                max_votes = max(vote_counts.values())
                top_targets = [tid for tid, cnt in vote_counts.items() if cnt == max_votes]
                
                # 투표 1위 대상에게 대세 따르기 보너스 50.0 이상 부여
                for target_id in top_targets:
                    self.belief[target_id, 3] += 60.0  # 대세를 따르는 강력한 가중치
            
            # 다수가 의심하는 사람의 마피아 점수 대폭 증가
            for target_id, count in vote_counts.items():
                # 표를 많이 받을수록 마피아 의심 증가
                self.belief[target_id, 3] += count * 20
        
        # 토론 내용 반영 (claim 정보)
        if 'claims' in game_status:
            claim_counts = {}
            for claimer_id, target_id in game_status['claims']:
                if target_id >= 0:
                    claim_counts[target_id] = claim_counts.get(target_id, 0) + 1
            
            # 여러 사람이 지목한 대상의 마피아 점수 증가
            for target_id, count in claim_counts.items():
                self.belief[target_id, 3] += count * 25
        
        # 자신의 belief는 유지
        self.belief[self.id, self.role] = 100.0
