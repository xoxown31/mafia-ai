from abc import ABC
from typing import List
import numpy as np
import random
import config


# Softmax 유틸리티 함수
def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax 함수: 점수(logits)를 확률 분포로 변환
    temperature: 낮을수록 결정론적, 높을수록 균등 분포
    """
    # 오버플로우 방지
    scores = scores / temperature
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    return exp_scores / np.sum(exp_scores)


class BaseCharacter(ABC):
    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        self.id = player_id
        self.role = role
        self.alive = True
        self.claimed_target = -1

        # Belief Matrix: (N x 4) - 각 플레이어가 각 직업일 것이라는 신뢰 점수
        # 열(Col): [0: 시민, 1: 경찰, 2: 의사, 3: 마피아]
        self.belief = np.zeros((config.PLAYER_COUNT, 4), dtype=np.float32)
        
        # 자신의 belief는 확실하게 설정
        self.belief[self.id, self.role] = 100.0

        # 게임 히스토리 추적
        self.voted_by_last_turn = []
        self.vote_history = [0] * config.PLAYER_COUNT
        self.investigated_players = set()  # 경찰이 조사한 플레이어 추적
        self.confirmed_mafia = set()  # 경찰이 확인한 마피아
        self.should_reveal = False  # 경찰이 공개할지 여부
        self.char_name = self.__class__.__name__

    def _get_alive_ids(
        self, players: List["BaseCharacter"], exclude_me: bool = True
    ) -> List[int]:
        """살아있는 플레이어 ID 목록 반환"""
        exclude_id = self.id if exclude_me else -1
        return [p.id for p in players if p.alive and p.id != exclude_id]

    def _select_target_softmax(
        self, 
        candidates: List[int], 
        scores: np.ndarray, 
        temperature: float = 0.1
    ) -> int:
        """
        Softmax 확률 기반으로 타겟 선택
        candidates: 후보 플레이어 ID 목록
        scores: 각 후보의 점수
        temperature: 확률 분포 조절 (낮을수록 결정론적)
        """
        if not candidates:
            return -1
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Softmax 확률 계산
        probs = softmax(scores, temperature)
        
        # 확률적 선택
        selected_idx = np.random.choice(len(candidates), p=probs)
        return candidates[selected_idx]

    def update_belief(self, game_status: dict):
        """
        게임 상태를 기반으로 belief matrix 업데이트
        자식 클래스에서 오버라이딩하여 성격별 로직 구현
        """
        # 기본 구현: 아무 것도 하지 않음
        pass

    def decide_claim(self, players: List["BaseCharacter"], current_day: int = 1) -> int:
        """
        낮 토론 시 지목할 플레이어 선택 (전략적 판단 포함)
        
        전략:
        1. 경찰: 마피아를 확인했을 때만 적극 주장 (확신 70% 이상)
        2. 시민: 확신이 있을 때만 주장 (마피아 의심 80% 이상)
        3. 마피아: 기회를 봐서 거짓 주장하거나 동조
        
        Args:
            players: 모든 플레이어 리스트
            current_day: 현재 게임 일수 (1부터 시작)
        
        Returns:
            선택된 플레이어 ID 또는 -1 (아무도 지목하지 않음)
        """
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            return -1
        
        # === 경찰의 전략: 마피아를 찾았을 때만 공개 ===
        if self.role == config.ROLE_POLICE:
            # 확인된 마피아가 있고, 그들이 살아있으면 적극 주장
            for mafia_id in self.confirmed_mafia:
                if players[mafia_id].alive:
                    self.should_reveal = True
                    return mafia_id
            
            # 마피아를 아직 못 찾았으면 조용히 있음 (초반에는 기권)
            if current_day <= 2:
                return -1
            
            # 후반부에는 가장 의심스러운 사람 지목
            max_suspicion = max(self.belief[pid, 3] for pid in alive_ids)
            if max_suspicion > 70:  # 확신이 있을 때만
                candidates = [pid for pid in alive_ids if self.belief[pid, 3] > 70]
                if candidates:
                    return candidates[0]
            return -1
        
        # === 시민의 전략: 매우 확신이 있을 때만 주장 ===
        elif self.role == config.ROLE_CITIZEN or self.role == config.ROLE_DOCTOR:
            # 초반에는 기권
            if current_day <= 2:
                return -1
            
            # 후반에는 확신이 매우 높을 때만 지목
            max_suspicion = max(self.belief[pid, 3] for pid in alive_ids)
            if max_suspicion > 80:  # 매우 확신할 때만
                candidates = [pid for pid in alive_ids if self.belief[pid, 3] > 80]
                if candidates:
                    return candidates[0]
            return -1
        
        # === 마피아의 전략: 전략적 거짓 주장 ===
        elif self.role == config.ROLE_MAFIA:
            # 다른 마피아 제외
            non_mafia_alive = [
                pid for pid in alive_ids 
                if players[pid].role != config.ROLE_MAFIA
            ]
            
            if not non_mafia_alive:
                return -1
            
            # 1일차: 조용히 있음 (정보 수집)
            if current_day == 1:
                return -1
            
            # 2일차 이후: 경찰로 의심되는 사람이 있으면 선제공격
            police_candidates = [
                pid for pid in non_mafia_alive 
                if self.belief[pid, 1] > 60  # 경찰로 의심
            ]
            if police_candidates:
                # 경찰을 거짓 주장으로 몰아가기
                return random.choice(police_candidates)
            
            # 아니면 랜덤하게 시민 한 명을 지목 (의심 분산)
            if random.random() < 0.4:  # 40% 확률로 주장
                return random.choice(non_mafia_alive)
            
            return -1
        
        return -1

    def decide_vote(self, players: List["BaseCharacter"], current_day: int = 1) -> int:
        """
        투표할 플레이어 선택 (전략적 판단 포함)
        
        전략:
        - 마피아 의심이 높은 사람에게 투표
        - 경찰이 공개 주장한 경우 우선순위 부여
        
        Args:
            players: 모든 플레이어 리스트
            current_day: 현재 게임 일수 (1부터 시작)
        
        Returns:
            선택된 플레이어 ID 또는 -1 (기권)
        """
        alive_ids = self._get_alive_ids(players, exclude_me=True)
        if not alive_ids:
            return -1
        
        # 마피아 의심 점수 계산
        mafia_scores = np.array([self.belief[pid, 3] for pid in alive_ids])
        
        # 가장 의심스러운 사람이 있으면 투표
        max_score = np.max(mafia_scores)
        
        # 초반(1-2일차)에는 확신이 매우 높을 때만 투표
        if current_day <= 2:
            if max_score > 70:
                candidates = [pid for pid, score in zip(alive_ids, mafia_scores) if score > 70]
                return self._select_target_softmax(candidates, 
                                                  np.array([self.belief[pid, 3] for pid in candidates]), 
                                                  temperature=0.1)
            return -1
        
        # 후반(3일차 이후)에는 적극적으로 투표
        if max_score > 30:  # 낮은 threshold
            return self._select_target_softmax(alive_ids, mafia_scores, temperature=0.1)
        
        return -1

    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        """밤 행동 결정 (직업별 분기)"""
        alive_ids = self._get_alive_ids(players, exclude_me=False)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            # 마피아: 경찰 공개자 우선 제거, 그 다음 위협적인 플레이어 선택
            candidates = [
                pid for pid in alive_ids 
                if pid != self.id and players[pid].role != config.ROLE_MAFIA
            ]
            
            if not candidates:
                return -1
            
            # 1순위: 경찰로 공개된 사람 (should_reveal이 true인 경찰)
            revealed_police = [
                pid for pid in candidates
                if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE
            ]
            if revealed_police:
                return revealed_police[0]  # 즉시 제거
            
            # 2순위: 경찰 or 의사로 강하게 의심되는 사람
            threat_scores = np.array([
                self.belief[pid, 1] * 1.5 + self.belief[pid, 2]  # 경찰에 더 높은 가중치
                for pid in candidates
            ])
            
            return self._select_target_softmax(candidates, threat_scores, temperature=0.1)

        elif current_role == config.ROLE_DOCTOR:
            # 의사: 경찰로 의심되는 사람 보호 (특히 공개한 경찰)
            candidates = [pid for pid in alive_ids]
            
            if not candidates:
                return -1
            
            # 공개한 경찰 우선 보호
            revealed_police = [
                pid for pid in candidates
                if players[pid].should_reveal and players[pid].role == config.ROLE_POLICE
            ]
            if revealed_police:
                return revealed_police[0]
            
            police_scores = np.array([
                self.belief[pid, 1] for pid in candidates
            ])
            
            return self._select_target_softmax(candidates, police_scores, temperature=0.1)

        elif current_role == config.ROLE_POLICE:
            # 경찰: 아직 조사하지 않은 사람 중 마피아 의심 높은 사람 조사
            candidates = [
                pid for pid in alive_ids 
                if pid != self.id and pid not in self.investigated_players
            ]
            
            if not candidates:
                # 모두 조사했으면 그냥 마피아 의심 높은 사람 반환
                candidates = [pid for pid in alive_ids if pid != self.id]
                if not candidates:
                    return -1
            
            mafia_scores = np.array([
                self.belief[pid, 3] for pid in candidates
            ])
            
            target = self._select_target_softmax(candidates, mafia_scores, temperature=0.1)
            self.investigated_players.add(target)
            return target

        return -1