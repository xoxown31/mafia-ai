import random
from typing import List
import config
from core.characters.base import BaseCharacter
import math
import random


class RationalCharacter(BaseCharacter):
    def _calculate_probs(self) -> List[float]:
        """[내부 함수] 현재 의심 점수들을 Softmax 확률로 변환"""
        scores = self.suspicion

        # 1. 오버플로우 방지: 점수 중 최대값을 구해서 모든 점수에서 뺌
        max_score = max(scores)

        # 2. 지수 함수(exp) 적용 (Temperature = 20.0 적용하여 분포 완화)
        T = 20.0
        exp_scores = [math.exp((s - max_score) / T) for s in scores]

        # 3. 전체 합으로 나누어 확률(0~1)로 정규화
        sum_exp = sum(exp_scores)
        probs = [e / sum_exp for e in exp_scores]

        return probs

    def get_suspicion_prob(self, target_id: int) -> float:
        """특정 플레이어에 대한 의심 확률 반환"""
        probs = self._calculate_probs()
        return probs[target_id]

    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        alive_indices = self._get_alive_ids(players, exclude_me=True)
        if not alive_indices:
            return -1

        # 1. 전체 확률 계산
        all_probs = self._calculate_probs()
        # 2. 생존자들의 확률만 추려냄 (가중치로 사용)
        alive_probs = [all_probs[i] for i in alive_indices]

        # (확률 기반이라도 의심도가 너무 낮으면 지목 안 함)
        current_max_score = max(self.suspicion[i] for i in alive_indices)
        if current_max_score < 10:
            return -1

        # 3. [핵심 변경] 가중치(weights)를 적용하여 랜덤 선택
        target = random.choices(alive_indices, weights=alive_probs, k=1)[0]

        return target

    def update_suspicion(self, speaker: "BaseCharacter", target_id: int):
        # [핵심] 결정 시 확률로 변환: 화자의 의심 확률을 확인
        speaker_prob = self.get_suspicion_prob(speaker.id)
        # 화자를 신뢰한다면(의심 확률 0.4 미만)
        if speaker_prob < 0.4:
            # 타겟의 의심 점수(Score)를 증가
            self.suspicion[target_id] += 10

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return self.decide_claim(players)

    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        all_probs = self._calculate_probs()

        if current_role == config.ROLE_MAFIA:
            # 타겟 후보: 살아있고, 동료 마피아가 아닌 사람
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]

            if not targets:
                # (혹시라도 쏠 대상이 없으면 - 보통 게임 종료 조건이라 이럴 일 없음)
                return alive_ids[0]

            # 2. 타겟 후보들의 확률(가중치)만 추출
            target_weights = [all_probs[i] for i in targets]

            # 3. [핵심 변경] 가중치 기반 랜덤 선택
            return random.choices(targets, weights=target_weights, k=1)[0]

        elif current_role == config.ROLE_DOCTOR:
            # 점수가 낮은 사람(신뢰하는 동료) 치료
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: self.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            return max(alive_ids, key=lambda i: self.suspicion[i])

        return -1
