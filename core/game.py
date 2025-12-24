from typing import List, Dict, Tuple
import config
import random


class Agent:
    def __init__(self):
        self.id = 0
        self.suspicion = []
        self.role = config.ROLE_CITIZEN  # 이제 숫자(0)로 초기화됨
        self.character = 0
        self.alive = True
        self.claimed_target = -1  # 주장 대상 초기화


class MafiaGame:
    """
    순수 마피아 게임 엔진
    """

    def __init__(self, log_file=None):
        self.players = []
        self.phase = config.PHASE_DAY_CLAIM
        self.day_count = 1
        self.alive_status = []
        self.vote_counts = []
        self.log_file = log_file

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> Dict:
        """게임을 초기화하고 초기 상태를 반환"""
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = config.PHASE_DAY_CLAIM
        self.players = [Agent() for _ in range(config.PLAYER_COUNT)]

        roles_to_assign = config.ROLES.copy()
        random.shuffle(roles_to_assign)

        role_names = {v: k.replace('ROLE_', '') for k, v in config.__dict__.items() if k.startswith('ROLE_')}

        for i, role_int in enumerate(roles_to_assign):
            self.players[i].role = role_int
            self.players[i].id = i
            self.players[i].suspicion = [0.5 for _ in range(config.PLAYER_COUNT)]
            self.players[i].claimed_target = -1
            self.players[i].alive = True
            role_name = role_names.get(role_int, "Unknown")
            self._log(f"플레이어 {i}: {role_name}")


        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]

        return self._get_game_status()

    def process_turn(self, action: int) -> Tuple[Dict, bool, bool]:
        # 승패 조건 먼저 체크
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win
            
        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        # 페이즈별 로직 실행
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

        # 턴 종료 후 승패 다시 체크
        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def morning_turn(self, ai_action: int):
        """낮 단계: 상태에 따라 적절한 처리 메서드 호출"""

        if self.phase == config.PHASE_DAY_CLAIM:
            self._process_day_claim(ai_action)
            self.phase = config.PHASE_DAY_DISCUSSION  # 다음 단계 전환

        elif self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion()
            self.phase = config.PHASE_DAY_VOTE  # 다음 단계 전환

        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote(ai_action)
            self.phase = config.PHASE_NIGHT  # 다음 단계 전환

    def _process_day_claim(self, ai_action: int):
        """[단계 1] 주장: 누굴 의심하는지 선언"""
        # AI 주장 반영
        if self.players[0].alive:
            self.players[0].claimed_target = ai_action
            self._log(f"  - AI(0)이(가) {ai_action}을(를) 지목했습니다.")

        # 봇들 주장 결정 (가장 의심하는 사람 지목)
        for p in self.players:
            if not p.alive or p.id == 0:
                continue

            max_suspicion = max(p.suspicion)
            max_indices = [
                i for i, suspicion in enumerate(p.suspicion) 
                if self.players[i].alive and suspicion == max_suspicion and i != p.id
            ]
            if max_indices:
                p.claimed_target = random.choice(max_indices)
                self._log(f"  - 플레이어 {p.id}이(가) {p.claimed_target}을(를) 지목했습니다.")


    def _process_day_discussion(self):
        """[단계 2] 토론: 다른 사람의 주장을 듣고 의심도 갱신"""
        self._log("  - 토론이 진행되어 의심도가 갱신됩니다.")
        player_count = len(self.players)

        for speaker_idx in range(player_count):
            speaker = self.players[speaker_idx]
            if not speaker.alive or speaker.claimed_target == -1:
                continue

            target_idx = speaker.claimed_target

            # 청자(Listener)들의 반응
            for listener_idx in range(player_count):
                listener = self.players[listener_idx]
                if speaker_idx == listener_idx or not listener.alive:
                    continue

                # 화자를 신뢰한다면(의심도 < 0.3), 화자의 주장에 동조
                if listener.suspicion[speaker_idx] < 0.3:
                    listener.suspicion[target_idx] = min(
                        1.0, listener.suspicion[target_idx] + 0.1
                    )

    def _process_day_vote(self, ai_action: int):
        """[단계 3] 투표: 최종 의심도를 바탕으로 처형 대상 결정"""
        player_count = len(self.players)
        self.vote_counts = [0] * player_count

        # AI 투표
        if self.players[0].alive:
            target = ai_action if 0 <= ai_action < player_count else 0
            self.vote_counts[target] += 1
            self._log(f"  - AI(0)이(가) {target}에게 투표했습니다.")

        # 봇 투표
        for p in self.players:
            if not p.alive or p.id == 0:
                continue

            max_suspicion = max(p.suspicion)
            max_indices = [
                i for i, suspicion in enumerate(p.suspicion) 
                if self.players[i].alive and suspicion == max_suspicion and i != p.id
            ]

            if max_indices:
                target = random.choice(max_indices)
                self.vote_counts[target] += 1
                self._log(f"  - 플레이어 {p.id}이(가) {target}에게 투표했습니다.")

        # 처형 집행
        max_votes = max(self.vote_counts)
        if max_votes > 0:
            executed_targets = [i for i, v in enumerate(self.vote_counts) if v == max_votes]
            executed_target = random.choice(executed_targets) # 동률일 경우 랜덤 선택
            self.players[executed_target].alive = False
            self._log(f"  - 투표 결과: {self.vote_counts}")
            self._log(f"  - {executed_target}번 플레이어가 처형되었습니다.")


            # 처형된 사람에 대한 의심도 초기화
            for p in self.players:
                p.suspicion[executed_target] = -1.0

        self._update_alive_status()

    def night_turn(self, ai_action: int):
        """밤 단계: 직업 행동 수행"""
        player_count = len(self.players)
        ai_player = self.players[0]

        # 1. 마피아 행동
        mafia_target = None
        mafia_list = [
            p for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        ]
        if mafia_list:
            if ai_player.role == config.ROLE_MAFIA and ai_player.alive:
                mafia_target = ai_action
                self._log(f"  - [마피아] AI(0)이(가) {mafia_target}을(를) 지목했습니다.")
            else:
                # AI가 마피아가 아닐 경우, 살아있는 마피아 중 첫번째가 행동
                shooter = next((p for p in mafia_list if p.id != 0), mafia_list[0])
                
                # 살아있고 마피아가 아닌 대상 중에서 가장 의심도가 높은 사람 선택
                valid_targets = [
                    i for i in range(player_count)
                    if self.players[i].alive and self.players[i].role != config.ROLE_MAFIA
                ]
                if valid_targets:
                    max_suspicion = max(shooter.suspicion[i] for i in valid_targets)
                    max_targets = [i for i in valid_targets if shooter.suspicion[i] == max_suspicion]
                    mafia_target = random.choice(max_targets)
                    self._log(f"  - [마피아] {shooter.id}이(가) {mafia_target}을(를) 지목했습니다.")

        # 2. 의사 행동
        doctor_target = None
        doctor_list = [
            p for p in self.players if p.role == config.ROLE_DOCTOR and p.alive
        ]
        for doctor in doctor_list:
            if doctor.id == 0: # AI가 의사일 경우
                doctor_target = ai_action
                self._log(f"  - [의사] AI(0)이(가) {doctor_target}을(를) 치료했습니다.")
            else:
                # 살아있는 사람 중 가장 의심도가 낮은(신뢰하는) 사람 치료
                valid_targets = [i for i in range(player_count) if self.players[i].alive]
                if valid_targets:
                    min_suspicion = min(doctor.suspicion[i] for i in valid_targets)
                    min_targets = [i for i in valid_targets if doctor.suspicion[i] == min_suspicion]
                    doctor_target = random.choice(min_targets)
                    self._log(f"  - [의사] {doctor.id}이(가) {doctor_target}을(를) 치료했습니다.")

        # 3. 경찰 행동
        police_list = [
            p for p in self.players if p.role == config.ROLE_POLICE and p.alive
        ]
        for police in police_list:
            if police.id == 0: # AI가 경찰일 경우
                target = ai_action
            else:
                # 살아있는 사람 중 가장 의심도가 높은 사람 조사
                valid_targets = [i for i in range(player_count) if self.players[i].alive]
                if valid_targets:
                    max_suspicion = max(police.suspicion[i] for i in valid_targets)
                    max_targets = [i for i in valid_targets if police.suspicion[i] == max_suspicion]
                    target = random.choice(max_targets)
                else:
                    continue
            
            target_role = "마피아" if self.players[target].role == config.ROLE_MAFIA else "시민"
            if police.id == 0:
                self._log(f"  - [경찰] AI(0)이(가) {target}을(를) 조사하여 '{target_role}'임을 확인했습니다.")
            else:
                self._log(f"  - [경찰] {police.id}이(가) {target}을(를) 조사하여 '{target_role}'임을 확인했습니다.")

            if self.players[target].role == config.ROLE_MAFIA:
                police.suspicion[target] = 1.0
            else:
                police.suspicion[target] = 0.0

        # 결과 정산
        if mafia_target is not None and mafia_target != doctor_target:
            self.players[mafia_target].alive = False
            self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
        elif mafia_target is not None and mafia_target == doctor_target:
            self._log(f"  - 의사가 {doctor_target}을(를) 살려냈습니다.")

        self._update_alive_status()

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
            # [수정 포인트] 이미 숫자이므로 변환 없이 그대로 내보냄
            "roles": [p.role for p in self.players],
            "id": 0,  # random.randint(0, config.PLAYER_COUNT - 1)
        }

    # 게임 종료 여부 및 승리 팀 확인
    def check_game_over(self) -> Tuple[bool, bool]:
        # 생존한 마피아와 시민 수 계산
        mafia_count = sum(
            1 for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        )
        citizen_count = sum(
            1 for p in self.players if p.role != config.ROLE_MAFIA and p.alive
        )
        
        is_over = False
        is_win = False

        # 1. 시민 팀 승리 조건 (마피아 전멸)
        if mafia_count == 0:
            am_i_citizen_team = self.players[0].role != config.ROLE_MAFIA
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            is_over, is_win = True, am_i_citizen_team

        # 2. 마피아 팀 승리 조건 (마피아 수 >= 시민 수)
        elif mafia_count >= citizen_count:
            am_i_mafia = self.players[0].role == config.ROLE_MAFIA
            self._log(f"\n게임 종료: 마피아 수가 시민 수와 같거나 많아졌습니다. 마피아 팀 승리!")
            is_over, is_win = True, am_i_mafia

        # 게임 계속
        return is_over, is_win
