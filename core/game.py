from typing import List, Dict, Tuple
import config


class Agent:
    def __init__(self):
        self.id = 0  # id
        self.suspicion = []  # 0~1 사이의 의심도
        self.role = config.ROLE_CITIZEN  # 역할
        self.character = 0  # 캐릭터 성격
        self.alive = True  # 생존여부


class MafiaGame:
    """
    순수 마피아 게임 엔진 (Gym/PyTorch 의존성 없음)
    """

    def __init__(self):
        self.players = []  # 플레이어 객체 리스트
        self.phase = "day"  # day, vote, night
        self.day_count = 1
        self.night_state = 0  # 죽은 사람 id
        self.vote = []  # 투표수
        self.ailve_number = 8

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

    def morning_turn(self):

        # 투표수 초기화
        self.vote = [0 for i in range(self.players.length)]

        # 주장 단계
        for i in range(self.players.length):
            argue = list.index(max(self.players[i].suspicion))
            for j in range(self.players.length):
                if i == j:
                    continue
                else:
                    if self.players[j].suspicion[i] < 0.3:
                        self.players[j].suspicion[argue] += 0.1

        # 투표
        for i in range(self.players.length):
            target = list.index(max(self.players[i].suspicion))
            self.vote[target] += 1

        result = list.index(max(self.vote))
        self.players[result].alive = False
        self.ailve_number -= 1

        for i in range(self.players.length):
            if i == result:
                continue
            else:
                self.players[i].suspicion[result] = -1

    def night_turn(self):
        # 경찰
        police = next((i for i, p in enumerate(self.players) if p.role == 1), None)
        police_target = list.index(max(self.players[police].suspicion))
        if self.players[police_target].role == 3:
            self.players[police].suspicion[police_target] = 1
        else:
            self.players[police].suspicion[police_target] = 0

        # 마피아
        mafia = [i for i, p in enumerate(self.players) if p.role == 3]
        mafia_zero = self.players[mafia[0]].suspicion
        mafia_one = self.players[mafia[1]].suspicion
        mafia_target = list.index(
            max(
                [i for i in range(len(mafia_zero))],
                key=lambda i: mafia_zero[i] + mafia_one[i],
            )
        )
        self.players[mafia_target].alive = False

        # 의사
        doctor = next((i for i, p in enumerate(self.players) if p.role == 2), None)
        self.players[doctor].alive = True

        self.day_count += 1

    def _get_game_status(self) -> Dict:
        """현재 게임 상태를 딕셔너리로 반환 (env에서 벡터화할 원본 데이터)"""
        return {
            "day": self.day_count,
            "phase": self.phase,
            "alive": [],  # TODO: 생존 여부 리스트
        }
