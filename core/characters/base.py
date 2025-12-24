from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
import config

# [수정] sigmoid 임포트 제거
# from core.utils import sigmoid


class BaseCharacter(ABC):
    """
    이제 이 클래스가 플레이어(Agent)이자 성격(Character)입니다.
    상태(Data)와 행동(Logic)을 모두 가집니다.
    """

    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        # [데이터/상태] 기존 Agent에 있던 것들
        self.id = player_id
        self.role = role
        self.alive = True
        self.claimed_target = -1

        # [수정] 의심도 관련: Logit 대신 직접 확률(0.0~1.0) 저장
        # 초기값 0.5 (중립)
        self.suspicion = [0.5 for _ in range(config.PLAYER_COUNT)]

        # 기억 관련
        self.voted_by_last_turn = []
        self.vote_history = [0] * config.PLAYER_COUNT

        # 로그 출력용 이름 (클래스 이름 사용)
        self.char_name = self.__class__.__name__

    # [수정] suspicion 프로퍼티 삭제됨 (위에서 self.suspicion 변수로 직접 선언)

    # [헬퍼 메서드]
    def _get_alive_ids(
        self, players: List["BaseCharacter"], exclude_me: bool = True
    ) -> List[int]:
        exclude_id = self.id if exclude_me else -1
        return [p.id for p in players if p.alive and p.id != exclude_id]

    # [추상 메서드]
    @abstractmethod
    def decide_claim(self, players: List["BaseCharacter"]) -> int:
        pass

    @abstractmethod
    def update_suspicion(self, speaker: "BaseCharacter", target_id: int):
        pass

    @abstractmethod
    def decide_vote(self, players: List["BaseCharacter"]) -> int:
        pass

    @abstractmethod
    def decide_night_action(
        self, players: List["BaseCharacter"], current_role: int
    ) -> int:
        pass
