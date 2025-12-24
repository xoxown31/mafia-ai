import random
from typing import List
import config
from core.characters.base import BaseCharacter

class RationalCharacter(BaseCharacter):
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        alive_indices = self._get_alive_ids(players, exclude_me=True)
        if not alive_indices:
            return -1

        # 점수가 가장 높은 사람이 확률도 가장 높으므로, max() 그대로 사용 가능
        max_score = max(self.suspicion[i] for i in alive_indices)
        candidates = [i for i in alive_indices if self.suspicion[i] == max_score]
        return random.choice(candidates)

    def update_suspicion(self, speaker: "BaseCharacter", target_id: int):
        # [핵심] 결정 시 확률로 변환: 화자의 의심 확률을 확인
        speaker_prob = self.get_suspicion_prob(speaker.id)
        
        # 화자를 신뢰한다면(의심 확률 0.4 미만)
        if speaker_prob < 0.4:
            # 타겟의 의심 점수(Score)를 증가
            self.suspicion[target_id] += 50

    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        return self.decide_claim(players)

    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]
            if not targets:
                return alive_ids[0]
            # 점수가 높은 사람(방해되는 시민) 제거
            return max(targets, key=lambda i: self.suspicion[i])

        elif current_role == config.ROLE_DOCTOR:
            # 점수가 낮은 사람(신뢰하는 동료) 치료
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: self.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            return max(alive_ids, key=lambda i: self.suspicion[i])

        return -1