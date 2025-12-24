from abc import ABC, abstractmethod
import random
from typing import List, TYPE_CHECKING
import config

if TYPE_CHECKING:
    from core.game import Agent  # 순환 참조 방지용


class BaseCharacter(ABC):
    """
    모든 성격(Character)의 기본이 되는 추상 클래스
    """

    @abstractmethod
    def decide_claim(self, agent: "Agent", players: List["Agent"]) -> int:
        """주장 단계에서 누구를 지목할지 결정"""
        pass

    @abstractmethod
    def update_suspicion(self, agent: "Agent", speaker: "Agent", target_id: int):
        """토론 단계에서 화자의 말을 듣고 의심도를 갱신"""
        pass

    @abstractmethod
    def decide_vote(self, agent: "Agent", players: List["Agent"]) -> int:
        """투표 단계에서 누구에게 투표할지 결정"""
        pass

    @abstractmethod
    def decide_night_action(
        self, agent: "Agent", players: List["Agent"], current_role: int
    ) -> int:
        """밤 단계에서 직업 행동 결정"""
        pass

    def _get_alive_ids(self, players: List["Agent"], exclude_me: int = -1) -> List[int]:
        """나를 제외한 생존자 ID 목록 반환 헬퍼"""
        return [p.id for p in players if p.alive and p.id != exclude_me]


# 따라쟁이
class CopyCat(BaseCharacter):
    def decide_claim(self, agent: "Agent", players: List["Agent"]) -> int:
        # 나를 투표했던 생존자 선택
        alive_attackers = [
            pid for pid in agent.voted_by_last_turn if players[pid].alive
        ]

        if alive_attackers:
            # 의심도 증가 총량 (x 값)
            RETALIATION_AMOUNT = 0.4

            # 룰 2: 투표한 사람 수만큼 나누어 적용
            penalty = RETALIATION_AMOUNT / len(alive_attackers)

            for pid in alive_attackers:
                # 룰 1: 의심도 증가 (최대 1.0)
                agent.suspicion[pid] = min(1.0, agent.suspicion[pid] + penalty)

        return super().decide_claim(agent, players)

    def update_suspicion(self, agent: "Agent", speaker: "Agent", target_id: int):
        # 화자를 신뢰(의심도 0.4 미만)하면 의심도 증가 (+0.1)
        if agent.suspicion[speaker.id] < 0.4:
            agent.suspicion[target_id] = min(1.0, agent.suspicion[target_id] + 0.1)

    def decide_vote(self, agent: "Agent", players: List["Agent"]) -> int:
        # 주장과 동일하게 가장 의심하는 사람 투표
        return self.decide_claim(agent, players)

    def decide_night_action(
        self, agent: "Agent", players: List["Agent"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            # 마피아: 동료가 아닌 사람 중 가장 의심도 높은 사람(방해되는 시민) 제거
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]
            if not targets:
                return alive_ids[0]
            # (단순화) 그냥 랜덤 혹은 가장 싫어하는 사람
            return max(targets, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_DOCTOR:
            # 의사: 가장 신뢰하는 사람(의심도 낮은 사람) 치료 (자신 포함 가능)
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            # 경찰: 가장 의심스러운 사람 조사
            return max(alive_ids, key=lambda i: agent.suspicion[i])

        return -1


# 원한주의자
class Grudger(BaseCharacter):
    def decide_claim(self, agent: "Agent", players: List["Agent"]) -> int:
        # 나를 투표했던 생존자 선택
        alive_attackers = [
            pid for pid in agent.voted_by_last_turn if players[pid].alive
        ]

        if alive_attackers:
            # 의심도 무조건 1.0
            RETALIATION_AMOUNT = 1.0

            for pid in alive_attackers:
                # 룰 1: 의심도 증가 (최대 1.0)
                agent.suspicion[pid] = min(
                    1.0, agent.suspicion[pid] + RETALIATION_AMOUNT
                )

        return super().decide_claim(agent, players)

    def update_suspicion(self, agent: "Agent", speaker: "Agent", target_id: int):
        # 화자를 신뢰(의심도 0.4 미만)하면 의심도 증가 (+0.1)
        if agent.suspicion[speaker.id] < 0.4:
            agent.suspicion[target_id] = min(1.0, agent.suspicion[target_id] + 0.1)

    def decide_vote(self, agent: "Agent", players: List["Agent"]) -> int:
        # 주장과 동일하게 가장 의심하는 사람 투표
        return self.decide_claim(agent, players)

    def decide_night_action(
        self, agent: "Agent", players: List["Agent"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            # 마피아: 동료가 아닌 사람 중 가장 의심도 높은 사람(방해되는 시민) 제거
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]
            if not targets:
                return alive_ids[0]
            # (단순화) 그냥 랜덤 혹은 가장 싫어하는 사람
            return max(targets, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_DOCTOR:
            # 의사: 가장 신뢰하는 사람(의심도 낮은 사람) 치료 (자신 포함 가능)
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            # 경찰: 가장 의심스러운 사람 조사
            return max(alive_ids, key=lambda i: agent.suspicion[i])

        return -1


# 너그러운 따라쟁이
class CopyKitten(BaseCharacter):
    def decide_claim(self, agent: "Agent", players: List["Agent"]) -> int:
        # 누적 투표 횟수가 2회 이상인 생존자(상습범) 찾기
        repeat_offenders = [
            pid
            for pid, count in enumerate(agent.vote_history)
            if count >= 2 and players[pid].alive
        ]

        if repeat_offenders:
            # 의심도 증가량 (x 값)
            RETALIATION_AMOUNT = 0.5

            for pid in repeat_offenders:
                # 룰 2 적용: 2회 이상 공격한 사람만 의심도 대폭 증가
                agent.suspicion[pid] = min(
                    1.0, agent.suspicion[pid] + RETALIATION_AMOUNT
                )

        return super().decide_claim(agent, players)

    def update_suspicion(self, agent: "Agent", speaker: "Agent", target_id: int):
        if agent.suspicion[speaker.id] < 0.5:
            agent.suspicion[target_id] = min(1.0, agent.suspicion[target_id] + 0.1)

    def decide_vote(self, agent: "Agent", players: List["Agent"]) -> int:
        return self.decide_claim(agent, players)

    def decide_night_action(
        self, agent: "Agent", players: List["Agent"], current_role: int
    ) -> int:
        alive_ids = self._get_alive_ids(players)
        if not alive_ids:
            return -1

        if current_role == config.ROLE_MAFIA:
            targets = [i for i in alive_ids if players[i].role != config.ROLE_MAFIA]
            if not targets:
                return alive_ids[0]
            return max(targets, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_DOCTOR:
            all_alive = [p.id for p in players if p.alive]
            return min(all_alive, key=lambda i: agent.suspicion[i])

        elif current_role == config.ROLE_POLICE:
            return max(alive_ids, key=lambda i: agent.suspicion[i])

        return -1
