from typing import List, Dict, Tuple
import config
import random

# Import RationalCharacter directly
from core.characters.rational import RationalCharacter
from core.characters.base import BaseCharacter


class MafiaGame:
    def __init__(self, log_file=None):
        # players 리스트에는 RationalCharacter가 들어갑니다.
        self.players: List[BaseCharacter] = []
        self.phase = config.PHASE_DAY_DISCUSSION  # Start with discussion phase
        self.day_count = 1
        self.alive_status = []
        self.vote_counts = []
        self.log_file = log_file
        self.final_vote = 0
        self.last_execution_result = None  # Track execution results
        self.last_night_result = None  # Track night results

    def _log(self, message):
        if self.log_file:
            self.log_file.write(message + "\n")

    def reset(self) -> Dict:
        self._log("\n--- 새 게임 시작 ---")
        self.day_count = 1
        self.phase = config.PHASE_DAY_DISCUSSION  # Start with discussion phase
        self.players = []
        self.last_execution_result = None
        self.last_night_result = None

        # 1. 플레이어 생성 (All use RationalCharacter)
        for i in range(config.PLAYER_COUNT):
            # Create rational agent for all players
            player = RationalCharacter(player_id=i)
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
            self._log(f"플레이어 {i}: {role_name} (Rational Agent)")

        self.alive_status = [True for _ in range(config.PLAYER_COUNT)]
        return self._get_game_status()

    def process_turn(self, reveal_role: int, target_id: int, accused_role: int) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        if self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion(reveal_role, target_id, accused_role)
            self.phase = config.PHASE_DAY_VOTE
        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote(target_id)
            self.phase = config.PHASE_NIGHT
        elif self.phase == config.PHASE_NIGHT:
            self.night_turn(target_id)
            self.phase = config.PHASE_DAY_DISCUSSION
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def _log_claim(self, player_id: int, reveal_role: int, target_id: int, accused_role: int):
        role_names = {
            config.ROLE_CITIZEN: "시민",
            config.ROLE_POLICE: "경찰",
            config.ROLE_DOCTOR: "의사",
            config.ROLE_MAFIA: "마피아"
        }
        
        player_name = f"AI({player_id})" if player_id == 0 else f"플레이어 {player_id}"
        
        if reveal_role != -1 and target_id != -1 and accused_role != -1:
            my_role_name = role_names.get(reveal_role, "알 수 없음")
            accused_role_name = role_names.get(accused_role, "알 수 없음")
            self._log(f"  - {player_name}이(가) 자신이 {my_role_name}이라 밝히며 {target_id}번이 {accused_role_name}이라고 지목했습니다.")
        elif reveal_role != -1:
            my_role_name = role_names.get(reveal_role, "알 수 없음")
            self._log(f"  - {player_name}이(가) 자신이 {my_role_name}이라고 밝혔습니다.")
        elif target_id != -1 and accused_role != -1:
            accused_role_name = role_names.get(accused_role, "알 수 없음")
            self._log(f"  - {player_name}이(가) {target_id}번이 {accused_role_name}이라고 지목했습니다.")
        elif target_id != -1:
            self._log(f"  - {player_name}이(가) {target_id}번을 지목했습니다.")
        else:
            self._log(f"  - {player_name}이(가) 침묵했습니다.")

    def _process_day_discussion(self, reveal_role: int, target_id: int, accused_role: int):
        self._log("  - 낮 토론: 플레이어들이 의견을 나누고 의심도를 갱신합니다.")
        
        structured_claims = []
        MAX_DEBATE_ROUNDS = 5
        
        for debate_round in range(MAX_DEBATE_ROUNDS):
            round_claims = []
            all_silent = True
            
            for p in self.players:
                if not p.alive:
                    continue
                
                if p.id == 0:
                    if reveal_role != -1 or target_id != -1 or accused_role != -1:
                        all_silent = False
                        claim_dict = {
                            "speaker_id": 0,
                            "type": "CLAIM",
                            "reveal_role": reveal_role,
                            "target_id": target_id,
                            "accused_role": accused_role
                        }
                        round_claims.append(claim_dict)
                        structured_claims.append(claim_dict)
                        
                        p.claimed_role = reveal_role
                        p.claimed_target = target_id
                        
                        self._log_claim(0, reveal_role, target_id, accused_role)
                    else:
                        p.claimed_role = -1
                        p.claimed_target = -1
                        if debate_round == 0:
                            self._log("  - AI(0)이(가) 침묵했습니다.")
                else:
                    claim_dict = p.decide_claim(self.players, self.day_count, structured_claims)
                    
                    if claim_dict["type"] == "CLAIM":
                        all_silent = False
                        claim_dict["speaker_id"] = p.id
                        
                        if "accused_role" not in claim_dict:
                            claim_dict["accused_role"] = -1
                        
                        round_claims.append(claim_dict)
                        structured_claims.append(claim_dict)
                        
                        p.claimed_role = claim_dict.get("reveal_role", -1)
                        p.claimed_target = claim_dict["target_id"]
                        
                        self._log_claim(p.id, claim_dict.get("reveal_role", -1), 
                                      claim_dict["target_id"], claim_dict.get("accused_role", -1))
                    else:
                        p.claimed_role = -1
                        p.claimed_target = -1
                        if debate_round == 0:
                            self._log(f"  - 플레이어 {p.id}이(가) 침묵했습니다.")
            
            if round_claims:
                game_status = {
                    'claims': round_claims,
                    'alive_players': [p.id for p in self.players if p.alive],
                    'day': self.day_count,
                    'execution_result': self.last_execution_result,
                    'night_result': self.last_night_result
                }
                
                for player in self.players:
                    if player.alive:
                        player.update_belief(game_status)
            
            if all_silent:
                if debate_round > 0:
                    self._log(f"  - [토론 종료] 모든 플레이어가 침묵하여 토론을 종료합니다. (라운드 {debate_round + 1})")
                break
        
        game_status = {
            'claims': structured_claims,
            'alive_players': [p.id for p in self.players if p.alive],
            'day': self.day_count,
            'execution_result': self.last_execution_result,
            'night_result': self.last_night_result
        }
        
        for player in self.players:
            if player.alive:
                player.update_belief(game_status)

    def _process_day_vote(self, target_id: int):
        player_count = len(self.players)
        self.vote_counts = [0] * player_count

        for p in self.players:
            p.voted_by_last_turn = []

        if self.players[0].alive:
            if target_id == -1 or not (0 <= target_id < player_count):
                self._log(f"  - AI(0)이(가) 투표를 기권했습니다.")
            else:
                self.vote_counts[target_id] += 1
                self._log(f"  - AI(0)이(가) {target_id}에게 투표했습니다.")
                self.players[target_id].voted_by_last_turn.append(0)
                self.players[target_id].vote_history[0] += 1

        # 봇들 투표
        for p in self.players:
            if not p.alive or p.id == 0:
                continue

            target = p.decide_vote(self.players, self.day_count)

            if target != -1:
                self.vote_counts[target] += 1
                self._log(f"  - 플레이어 {p.id}이(가) {target}에게 투표했습니다.")

                self.players[target].voted_by_last_turn.append(p.id)
                self.players[target].vote_history[p.id] += 1
            else:
                self._log(f"  - 플레이어 {p.id}이(가) 투표를 기권했습니다.")

        self._log(f"  - 최종 투표 집계: {self.vote_counts}")

        # 처형 집행
        max_votes = max(self.vote_counts)
        self.final_vote = 0
        if max_votes > 0:
            executed_targets = [
                i for i, v in enumerate(self.vote_counts) if v == max_votes
            ]

            # === v2.2: TIE-BREAKER LOGIC ===
            # If multiple players tied for most votes, skip execution
            if len(executed_targets) > 1:
                self._log(f"  - 투표 동률 발생: {executed_targets}")
                self._log(f"  - 처형이 무산되었습니다. (Voting Tie)")
                self.last_execution_result = None
            else:
                # Only one player with max votes - proceed with execution
                executed_target = executed_targets[0]

                # Collect voters who voted for the executed target
                voters_for_target = []
                for i in range(len(self.players)):
                    if self.players[i].alive and executed_target in self.players[i].voted_by_last_turn:
                        continue
                    if i < len(self.players) and self.vote_counts[executed_target] > 0:
                        # Check who voted for this target
                        if executed_target in self.players[executed_target].voted_by_last_turn:
                            continue
                # Actually collect voters from voted_by_last_turn of the target
                voters_for_target = self.players[executed_target].voted_by_last_turn.copy()

                # AI 투표
                if self.players[0].alive:
                    # 마피아 의심 점수가 양수면 찬성
                    if self.players[0].belief[executed_target, 3] > 0:
                        self.final_vote += 1
                    else:
                        self.final_vote -= 1

                for p in self.players:
                    if not p.alive or p.id == 0:
                        continue
                    # 봇들 투표 집계
                    if p.belief[executed_target, 3] > 0:
                        self.final_vote += 1
                    else:
                        self.final_vote -= 1

                team_alignment = None
                if self.final_vote > 0:
                    self.players[executed_target].alive = False
                    self._log(f"  - 투표 결과: 찬성 {self.final_vote}표")
                    self._log(f"  - {executed_target}번 플레이어가 처형되었습니다.")
                    
                    # === ROLE REVEAL: Show team alignment (Citizen Team vs Mafia Team) ===
                    executed_role = self.players[executed_target].role
                    if executed_role == config.ROLE_MAFIA:
                        team_alignment = "MAFIA"
                        self._log(f"  - [공개] {executed_target}번은 마피아 팀이었습니다!")
                    else:
                        team_alignment = "CITIZEN"
                        self._log(f"  - [공개] {executed_target}번은 시민 팀이었습니다.")
                    
                    # Include vote log in execution result
                    vote_log = {'voters': voters_for_target, 'target': executed_target}
                    self.last_execution_result = (executed_target, team_alignment, self.day_count, vote_log)
                else:
                    self._log(f"  - 투표 결과: 찬성 {self.final_vote}표 (과반 미달)")
                    self._log(f"  - {executed_target}번 플레이어는 처형되지 않았습니다.")
                    self.last_execution_result = None

        self._update_alive_status()

    def night_turn(self, target_id: int):
        mafia_target = None
        doctor_target = None

        mafia_list = [
            p for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        ]
        if mafia_list:
            if self.players[0].role == config.ROLE_MAFIA and self.players[0].alive:
                mafia_target = target_id
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

        doctor_list = [
            p for p in self.players if p.role == config.ROLE_DOCTOR and p.alive
        ]
        for doctor in doctor_list:
            t = -1
            if doctor.id == 0:
                t = target_id
                self._log(f"  - [의사] AI(0)이(가) {t}을(를) 치료했습니다.")
            else:
                t = doctor.decide_night_action(self.players, config.ROLE_DOCTOR)
                self._log(f"  - [의사] {doctor.id}이(가) {t}을(를) 치료했습니다.")
            doctor_target = t

        police_list = [
            p for p in self.players if p.role == config.ROLE_POLICE and p.alive
        ]
        for police in police_list:
            t = -1
            if police.id == 0:
                t = target_id
            else:
                t = police.decide_night_action(self.players, config.ROLE_POLICE)

            if t != -1:
                role_str = (
                    "마피아"
                    if self.players[t].role == config.ROLE_MAFIA
                    else "시민"
                )
                prefix = "[경찰] AI(0)" if police.id == 0 else f"[경찰] {police.id}"
                self._log(
                    f"  - {prefix}이(가) {t}을(를) 조사하여 '{role_str}'임을 확인했습니다."
                )

                if self.players[t].role == config.ROLE_MAFIA:
                    police.belief[t, 3] = 100
                    police.belief[t, 0] = -100
                    police.confirmed_mafia.add(t)
                else:
                    police.belief[t, 3] = -100
                    police.belief[t, 0] = 50

        no_death = False
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
                no_death = False
            else:
                self._log(f"  - 의사가 {doctor_target}을(를) 살려냈습니다.")
                no_death = True
        
        self.last_night_result = {
            'no_death': no_death,
            'last_healed': doctor_target if doctor_target is not None else -1
        }

        self._update_alive_status()

    def _update_alive_status(self):
        self.alive_status = [1 if p.alive else 0 for p in self.players]

    def _get_game_status(self) -> Dict:
        phase_map = {
            config.PHASE_DAY_DISCUSSION: 0,
            config.PHASE_DAY_VOTE: 1,
            config.PHASE_NIGHT: 2,
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

        if self.day_count > config.MAX_DAYS:
            self._log(f"\n게임 종료: {config.MAX_DAYS}일이 지나 무승부입니다!")
            return True, False

        if mafia_count == 0:
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            return True, self.players[0].role != config.ROLE_MAFIA
        elif mafia_count >= citizen_count:
            self._log(f"\n게임 종료: 마피아 승리!")
            return True, self.players[0].role == config.ROLE_MAFIA

        return False, False
