from typing import List, Dict, Tuple
import config
import random

# [수정] create_player 함수 임포트
from core.characters import create_player, BaseCharacter


class MafiaGame:
    def __init__(self, log_file=None):
        # players 리스트에는 이제 BaseCharacter(의 자식들)가 들어갑니다.
        self.players: List[BaseCharacter] = []
        self.phase = config.PHASE_DAY_CLAIM
        self.day_count = 1
        self.alive_status = []
        self.vote_counts = []
        self.log_file = log_file
        self.final_vote = 0

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> Dict:
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = config.PHASE_DAY_CLAIM
        self.players = []

        possible_char_ids = [
            config.CHAR_RATIONAL,
            config.CHAR_COPYCAT,
            config.CHAR_GRUDGER,
            config.CHAR_COPYKITTEN,
        ]

        # 1. 플레이어 생성 (Agent 없이 바로 Character 생성)
        for i in range(config.PLAYER_COUNT):
            char_id = random.choice(possible_char_ids)
            # 팩토리 함수를 통해 생성 (id 부여)
            player = create_player(char_id=char_id, player_id=i)
            self.players.append(player)

        # 2. 역할 할당
        roles_to_assign = config.ROLES.copy()
        random.shuffle(roles_to_assign)

        role_names = {
            v: k.replace("ROLE_", "")
            for k, v in config.__dict__.items()
            if k.startswith("ROLE_")
        }

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int

            # 로그 출력
            role_name = role_names.get(role_int, "Unknown")
            char_name = self.players[i].char_name
            self._log(f"플레이어 {i}: {role_name} ({char_name})")

        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]
        return self._get_game_status()

    def process_turn(self, action: int) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        if self.phase == config.PHASE_DAY_CLAIM:
            self._process_day_claim(action)
            self.phase = config.PHASE_DAY_DISCUSSION
        elif self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion()
            self.phase = config.PHASE_DAY_VOTE
        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote(action)
            self.phase = config.PHASE_NIGHT
        elif self.phase == config.PHASE_NIGHT:
            self.night_turn(action)
            self.phase = config.PHASE_DAY_CLAIM
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def _process_day_claim(self, ai_action: int):
        # AI(0번) 처리
        if self.players[0].alive:
            self.players[0].claimed_target = ai_action
            self._log(f"  - AI(0)이(가) {ai_action}을(를) 지목했습니다.")

        # 봇들 처리 (Agent 없이 바로 메서드 호출)
        for p in self.players:
            if not p.alive or p.id == 0:
                continue

            # p.decide_claim(self.players) 호출 (인자로 players 리스트만 넘김)
            target = p.decide_claim(self.players)

            if target != -1:
                p.claimed_target = target
                self._log(f"  - 플레이어 {p.id}이(가) {target}을(를) 지목했습니다.")
            else:
                p.claimed_target = -1
                self._log(f"  - 플레이어 {p.id}이(가) 아무도 지목하지 않았습니다.")

    def _process_day_discussion(self):
        self._log("  - 토론이 진행되어 의심도가 갱신됩니다.")
        for speaker in self.players:
            if not speaker.alive or speaker.claimed_target == -1:
                continue

            target_idx = speaker.claimed_target

            for listener in self.players:
                if speaker.id == listener.id or not listener.alive:
                    continue

                # listener.update_suspicion 호출
                listener.update_suspicion(speaker, target_idx)

    def _process_day_vote(self, ai_action: int):
        player_count = len(self.players)
        self.vote_counts = [0] * player_count

        # 투표 기록 초기화
        for p in self.players:
            p.voted_by_last_turn = []

        # AI 투표
        if self.players[0].alive:
            target = ai_action if 0 <= ai_action < player_count else 0
            self.vote_counts[target] += 1
            self._log(f"  - AI(0)이(가) {target}에게 투표했습니다.")

            self.players[target].voted_by_last_turn.append(0)
            self.players[target].vote_history[0] += 1

        # 봇들 투표
        for p in self.players:
            if not p.alive or p.id == 0:
                continue

            target = p.decide_vote(self.players)

            if target != -1:
                self.vote_counts[target] += 1
                self._log(f"  - 플레이어 {p.id}이(가) {target}에게 투표했습니다.")

                self.players[target].voted_by_last_turn.append(p.id)
                self.players[target].vote_history[p.id] += 1

        self._log(f"  - 최종 투표 집계: {self.vote_counts}")

        # 처형 집행
        max_votes = max(self.vote_counts)
        self.final_vote = 0
        if max_votes > 0:
            executed_targets = [
                i for i, v in enumerate(self.vote_counts) if v == max_votes
            ]

            executed_target = random.choice(executed_targets)

            # AI 투표
            if self.players[0].alive:
                if self.players[0].suspicion[executed_target] > 0:
                    self.final_vote += 1
                else:
                    self.final_vote -= 1

            for p in self.players:
                if not p.alive or p.id == 0:
                    continue
                # 봇들 투표 집계
                if p.suspicion[executed_target] > 0:
                    self.final_vote += 1
                else:
                    self.final_vote -= 1

            if self.final_vote > 0:
                self.players[executed_target].alive = False
                self._log(f"  - 투표 결과: 찬성 {self.final_vote}표")
                self._log(f"  - {executed_target}번 플레이어가 처형되었습니다.")
            elif self.final_vote == 0:
                self._log(f"  - 투표 결과: 찬반 동수로 처형이 무산되었습니다.")
            else:
                self._log(f"  - 투표 결과: 반대 {abs(self.final_vote)}표")

            for p in self.players:
                # 죽은 사람 의심도 초기화 (Logit을 매우 낮게 설정)
                p.suspicion[executed_target] = -100

        self._update_alive_status()

    def night_turn(self, ai_action: int):
        mafia_target = None
        doctor_target = None

        # 1. 마피아
        mafia_list = [
            p for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        ]
        if mafia_list:
            if self.players[0].role == config.ROLE_MAFIA and self.players[0].alive:
                mafia_target = ai_action
                self._log(
                    f"  - [마피아] AI(0)이(가) {mafia_target}을(를) 지목했습니다."
                )
            else:
                shooter = mafia_list[0]
                mafia_target = shooter.decide_night_action(
                    self.players, config.ROLE_MAFIA
                )
                self._log(
                    f"  - [마피아] {shooter.id}이(가) {mafia_target}을(를) 지목했습니다."
                )

        # 2. 의사
        doctor_list = [
            p for p in self.players if p.role == config.ROLE_DOCTOR and p.alive
        ]
        for doctor in doctor_list:
            target = -1
            if doctor.id == 0:
                target = ai_action
                self._log(f"  - [의사] AI(0)이(가) {target}을(를) 치료했습니다.")
            else:
                target = doctor.decide_night_action(self.players, config.ROLE_DOCTOR)
                self._log(f"  - [의사] {doctor.id}이(가) {target}을(를) 치료했습니다.")
            doctor_target = target

        # 3. 경찰
        police_list = [
            p for p in self.players if p.role == config.ROLE_POLICE and p.alive
        ]
        for police in police_list:
            target = -1
            if police.id == 0:
                target = ai_action
            else:
                target = police.decide_night_action(self.players, config.ROLE_POLICE)

            if target != -1:
                role_str = (
                    "마피아"
                    if self.players[target].role == config.ROLE_MAFIA
                    else "시민"
                )
                prefix = "[경찰] AI(0)" if police.id == 0 else f"[경찰] {police.id}"
                self._log(
                    f"  - {prefix}이(가) {target}을(를) 조사하여 '{role_str}'임을 확인했습니다."
                )

                if self.players[target].role == config.ROLE_MAFIA:
                    police.suspicion[target] = 100  # 확신
                else:
                    police.suspicion[target] = -100  # 확신

        # 결과 정산
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
            else:
                self._log(f"  - 의사가 {doctor_target}을(를) 살려냈습니다.")

        self._update_alive_status()

    # (이하 _update_alive_status, _get_game_status, check_game_over 메서드는 Agent 참조만 수정해서 사용)
    def _update_alive_status(self):
        self.alive_status = [1 if p.alive else 0 for p in self.players]

    def _get_game_status(self) -> Dict:
        phase_map = {
            config.PHASE_DAY_CLAIM: 0,
            config.PHASE_DAY_DISCUSSION: 1,
            config.PHASE_DAY_VOTE: 2,
            config.PHASE_NIGHT: 3,
        }
        return {
            "day": self.day_count,
            "phase": phase_map.get(self.phase, 0),
            "alive_status": self.alive_status,
            "roles": [p.role for p in self.players],
            "id": 0,
        }

    def check_game_over(self) -> Tuple[bool, bool]:
        mafia_count = sum(
            1 for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        )
        citizen_count = sum(
            1 for p in self.players if p.role != config.ROLE_MAFIA and p.alive
        )

        if mafia_count == 0:
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            return True, self.players[0].role != config.ROLE_MAFIA
        elif mafia_count >= citizen_count:
            self._log(f"\n게임 종료: 마피아 승리!")
            return True, self.players[0].role == config.ROLE_MAFIA

        return False, False
