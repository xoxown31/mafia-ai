from typing import List
import numpy as np
from core.characters.base import BaseCharacter
import config


class Analyst(BaseCharacter):
    """
    냉철한 분석가 (Analyst)
    - 투표 패턴 분석에 집중
    - 감정을 배제하고 논리적으로 판단
    - 일관성 없는 행동을 하는 플레이어를 의심
    """

    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        super().__init__(player_id, role)
        # 각 플레이어의 투표 패턴 추적
        self.player_vote_patterns = [[] for _ in range(config.PLAYER_COUNT)]

    def update_belief(self, game_status: dict):
        """
        투표 패턴 분석:
        - 일관성 없는 투표를 하는 사람 의심
        - 항상 생존자만 찍는 사람 의심 (마피아 가능성)
        - 소수 의견을 지속적으로 내는 사람 분석
        """
        if 'vote_results' in game_status:
            # 투표 패턴 기록
            for voter_id, target_id in game_status['vote_results']:
                if target_id >= 0:
                    self.player_vote_patterns[voter_id].append(target_id)
            
            # 패턴 분석
            for pid in range(config.PLAYER_COUNT):
                if pid == self.id or not self.player_vote_patterns[pid]:
                    continue
                
                pattern = self.player_vote_patterns[pid]
                
                # 1. 투표 변동성이 높은 사람 (일관성 없음) - 마피아 의심
                if len(pattern) >= 3:
                    unique_targets = len(set(pattern[-3:]))
                    if unique_targets == 3:  # 최근 3번 모두 다른 사람 투표
                        self.belief[pid, 3] += 40  # 일관성 없는 행동 강하게 의심
                
                # 2. 항상 같은 사람만 찍는 사람 (집착) - 분석 필요
                if len(pattern) >= 2:
                    if len(set(pattern)) == 1:
                        # 특정 타겟에 집착 - 중립적 분석
                        target = pattern[0]
                        # 그 타겟이 실제로 의심스러운지 확인
                        if self.belief[target, 3] < 20:
                            # 별로 의심스럽지 않은 사람을 계속 찍음 = 의심스러움
                            self.belief[pid, 3] += 35
        
        # 발언 분석
        if 'claims' in game_status:
            claim_counts = {}
            for claimer_id, target_id in game_status['claims']:
                if target_id >= 0:
                    claim_counts[target_id] = claim_counts.get(target_id, 0) + 1
            
            # 고립된 의견을 내는 사람 분석
            for claimer_id, target_id in game_status['claims']:
                if target_id >= 0 and claim_counts[target_id] == 1:
                    # 혼자만 의심하는 경우 - 의심
                    self.belief[claimer_id, 3] += 30
        
        # 자신의 belief 확실하게 유지
        self.belief[self.id, self.role] = 100.0
