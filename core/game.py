from typing import List, Dict, Tuple
import config

class MafiaGame:
    """
    순수 마피아 게임 엔진 (Gym/PyTorch 의존성 없음)
    """
    def __init__(self):
        self.players = []  # 플레이어 객체 리스트
        self.phase = "day" # day, vote, night
        self.day_count = 1
        
    def reset(self) -> Dict:
        """게임을 초기화하고 초기 상태를 반환"""
        self.day_count = 1
        self.phase = "day"
        # TODO: 직업 랜덤 분배 및 플레이어 초기화
        return self._get_game_status()

    def process_turn(self, action: int) -> Tuple[Dict, bool, bool]:
        """
        한 턴(혹은 한 단계)을 진행
        Returns: (game_status, is_game_over, is_win)
        """
        # TODO: AI 행동 처리 -> RBA 행동 처리 -> 결과 정산
        return self._get_game_status(), False, False

    def _get_game_status(self) -> Dict:
        """현재 게임 상태를 딕셔너리로 반환 (env에서 벡터화할 원본 데이터)"""
        return {
            "day": self.day_count,
            "phase": self.phase,
            "alive": [],  # TODO: 생존 여부 리스트
        }