from typing import List, Dict, Tuple
import config
import random

# Import RationalCharacter directly
from core.agent.rational import RationalCharacter
from core.agent.base import BaseCharacter


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

        self.ai_id = random.randint(0, config.PLAYER_COUNT - 1)
        self._log(f"[System] 이번 게임의 AI 플레이어는 {self.ai_id}번 입니다.")

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

    def process_turn(
        self, action: int, ai_claim_role: int = -1
    ) -> Tuple[Dict, bool, bool]:
        is_over, is_win = self.check_game_over()
        if is_over:
            return self._get_game_status(), is_over, is_win

        self._log(f"\n[Day {self.day_count} | {self.phase}]")

        # Merged phases: PHASE_DAY_CLAIM removed
        if self.phase == config.PHASE_DAY_DISCUSSION:
            self._process_day_discussion(action, ai_claim_role)  # ai_claim_role 전달
            self.phase = config.PHASE_DAY_VOTE
        elif self.phase == config.PHASE_DAY_VOTE:
            self._process_day_vote(action)
            self.phase = config.PHASE_NIGHT
        elif self.phase == config.PHASE_NIGHT:
            self.night_turn(action)
            self.phase = config.PHASE_DAY_DISCUSSION
            self.day_count += 1

        is_over, is_win = self.check_game_over()
        return self._get_game_status(), is_over, is_win

    def _process_day_discussion(self, ai_action: int, ai_claim_role: int = -1):
        """Merged phase: Claims and discussion happen together with UNLIMITED DEBATE LOOP
        ai_claim_role: AI가 주장하는 역할 (-1: 주장 없음, 0-3: 역할)
        """
        self._log("  - 낮 토론: 플레이어들이 의견을 나누고 의심도를 갱신합니다.")
        structured_claims = []  # Store all claims in new dictionary format

        # === Step 1: AI 발언 처리 (self.ai_id 사용) ===
        if self.players[self.ai_id].alive:
            prefix = f"  - [AI 주장] AI({self.ai_id})가"  # 로그 포맷 변경

            if ai_claim_role != -1:
                role_names = {
                    config.ROLE_POLICE: "경찰",
                    config.ROLE_DOCTOR: "의사",
                    config.ROLE_CITIZEN: "시민",
                    config.ROLE_MAFIA: "마피아",
                }
                role_name = role_names.get(ai_claim_role, "알 수 없음")

                if ai_action != -1 and 0 <= ai_action < len(self.players):
                    assertion = (
                        "CONFIRMED_MAFIA"
                        if ai_claim_role == config.ROLE_POLICE
                        else "SUSPECT"
                    )
                    ai_claim = {
                        "speaker_id": self.ai_id,  # [변경] 0 -> self.ai_id
                        "type": "CLAIM",
                        "reveal_role": ai_claim_role,
                        "target_id": ai_action,
                        "assertion": assertion,
                    }
                    structured_claims.append(ai_claim)
                    self.players[self.ai_id].claimed_target = ai_action
                    self.players[self.ai_id].claimed_role = ai_claim_role
                    self._log(
                        f"{prefix} {role_name}이라 밝히며 {ai_action}을(를) 지목했습니다."
                    )
                else:
                    ai_claim = {
                        "speaker_id": self.ai_id,  # [변경]
                        "type": "CLAIM",
                        "reveal_role": ai_claim_role,
                        "target_id": -1,
                        "assertion": "NONE",
                    }
                    structured_claims.append(ai_claim)
                    self.players[self.ai_id].claimed_target = -1
                    self.players[self.ai_id].claimed_role = ai_claim_role
                    self._log(f"{prefix} 자신이 {role_name}이라고 밝혔습니다.")
            elif ai_action != -1 and 0 <= ai_action < len(self.players):
                ai_claim = {
                    "speaker_id": self.ai_id,  # [변경]
                    "type": "CLAIM",
                    "reveal_role": -1,
                    "target_id": ai_action,
                    "assertion": "SUSPECT",
                }
                structured_claims.append(ai_claim)
                self.players[self.ai_id].claimed_target = ai_action
                self.players[self.ai_id].claimed_role = -1
                self._log(f"  - AI({self.ai_id})가 {ai_action}을(를) 지목했습니다.")
            else:
                self.players[self.ai_id].claimed_target = -1
                self.players[self.ai_id].claimed_role = -1
                self._log(f"  - AI({self.ai_id})가 아무 주장도 하지 않았습니다.")

        # === Step 2: Bots debate until consensus (UNLIMITED LOOP) ===
        MAX_DEBATE_ROUNDS = 5

        for debate_round in range(MAX_DEBATE_ROUNDS):
            round_claims = []  # Claims made in this debate round
            all_silent = True  # Track if all bots are silent this round

            # 봇들 처리 (Player 1~7)
            for p in self.players:
                if not p.alive or p.id == self.ai_id:
                    continue

                # Pass current discussion context for reactive claims
                claim_dict = p.decide_claim(
                    self.players, self.day_count, structured_claims
                )

                if claim_dict["type"] == "CLAIM":
                    all_silent = False
                    # Add speaker_id to the claim
                    claim_dict["speaker_id"] = p.id
                    round_claims.append(claim_dict)
                    structured_claims.append(claim_dict)  # Add to global claims

                    # Set claimed_target for backwards compatibility
                    p.claimed_target = claim_dict["target_id"]

                    # Update claimed_role
                    reveal_role = claim_dict.get("reveal_role", -1)
                    if reveal_role != -1:
                        p.claimed_role = reveal_role  # 역할을 주장함
                    else:
                        p.claimed_role = -1  # 역할 주장 없음

                    # Enhanced logging based on claim type
                    target_id = claim_dict["target_id"]
                    reveal_role = claim_dict["reveal_role"]
                    assertion = claim_dict["assertion"]

                    if reveal_role != -1:
                        role_names = {
                            config.ROLE_POLICE: "경찰",
                            config.ROLE_DOCTOR: "의사",
                            config.ROLE_CITIZEN: "시민",
                            config.ROLE_MAFIA: "마피아",
                        }
                        role_name = role_names.get(reveal_role, "알 수 없음")

                        if assertion == "CONFIRMED_MAFIA":
                            self._log(
                                f"  - [중요] 플레이어 {p.id}이(가) {role_name}이라 밝히며 "
                                f"{target_id}이(가) 마피아라고 확정 주장합니다!"
                            )
                        elif assertion == "CONFIRMED_CITIZEN":
                            self._log(
                                f"  - [중요] 플레이어 {p.id}이(가) {role_name}이라 밝히며 "
                                f"{target_id}이(가) 시민이라고 확정 주장합니다!"
                            )
                        else:
                            self._log(
                                f"  - 플레이어 {p.id}이(가) {role_name}이라 밝히며 "
                                f"{target_id}을(를) 의심합니다."
                            )
                    else:
                        if assertion == "CONFIRMED_MAFIA":
                            self._log(
                                f"  - 플레이어 {p.id}이(가) {target_id}을(를) 마피아로 확신합니다!"
                            )
                        elif assertion == "CONFIRMED_CITIZEN":
                            self._log(
                                f"  - 플레이어 {p.id}이(가) {target_id}을(를) 시민으로 확신합니다!"
                            )
                        else:
                            self._log(
                                f"  - 플레이어 {p.id}이(가) {target_id}을(를) 의심합니다."
                            )
                else:
                    p.claimed_target = -1
                    p.claimed_role = -1  # 침묵
                    if debate_round == 0:  # Only log on first round
                        self._log(
                            f"  - 플레이어 {p.id}이(가) 아무 주장도 하지 않았습니다."
                        )

            # Update beliefs after each debate round (Real-time reaction)
            if round_claims:
                game_status = {
                    "claims": round_claims,  # Only new claims from this round
                    "alive_players": [p.id for p in self.players if p.alive],
                    "day": self.day_count,
                    "execution_result": self.last_execution_result,
                    "night_result": self.last_night_result,
                }

                for player in self.players:
                    if player.alive:
                        player.update_belief(game_status)

            # End debate if all bots are silent
            if all_silent:
                if debate_round > 0:
                    self._log(
                        f"  - [토론 종료] 모든 플레이어가 침묵하여 토론을 종료합니다. (라운드 {debate_round + 1})"
                    )
                break

        # === Step 3: Final belief update with all accumulated claims ===
        game_status = {
            "claims": structured_claims,  # All claims from entire discussion
            "alive_players": [p.id for p in self.players if p.alive],
            "day": self.day_count,
            "execution_result": self.last_execution_result,
            "night_result": self.last_night_result,
        }

        # Final sync for all players
        for player in self.players:
            if player.alive:
                player.update_belief(game_status)

    def _process_day_vote(self, ai_action: int):
        player_count = len(self.players)
        self.vote_counts = [0] * player_count

        # 투표 기록 초기화
        for p in self.players:
            p.voted_by_last_turn = []

        # [변경] AI 투표 처리 (self.ai_id 사용)
        if self.players[self.ai_id].alive:
            if ai_action == -1:
                self._log(f"  - AI({self.ai_id})가 투표를 기권했습니다.")
            else:
                target = (
                    ai_action if 0 <= ai_action < player_count else self.ai_id
                )  # 예외처리
                self.vote_counts[target] += 1
                self._log(f"  - AI({self.ai_id})가 {target}에게 투표했습니다.")

                self.players[target].voted_by_last_turn.append(self.ai_id)
                self.players[target].vote_history[self.ai_id] += 1

        # 봇들 투표
        for p in self.players:
            if not p.alive or p.id == self.ai_id:
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
                    if (
                        self.players[i].alive
                        and executed_target in self.players[i].voted_by_last_turn
                    ):
                        continue
                    if i < len(self.players) and self.vote_counts[executed_target] > 0:
                        # Check who voted for this target
                        if (
                            executed_target
                            in self.players[executed_target].voted_by_last_turn
                        ):
                            continue
                # Actually collect voters from voted_by_last_turn of the target
                voters_for_target = self.players[
                    executed_target
                ].voted_by_last_turn.copy()

                # AI 투표
                if self.players[self.ai_id].alive:
                    if self.players[self.ai_id].belief[executed_target, 3] > 0:
                        self.final_vote += 1
                    else:
                        self.final_vote -= 1

                for p in self.players:
                    if not p.alive or p.id == self.ai_id:  # [변경]
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
                        self._log(
                            f"  - [공개] {executed_target}번은 마피아 팀이었습니다!"
                        )
                    else:
                        team_alignment = "CITIZEN"
                        self._log(
                            f"  - [공개] {executed_target}번은 시민 팀이었습니다."
                        )

                    # Include vote log in execution result
                    vote_log = {"voters": voters_for_target, "target": executed_target}
                    self.last_execution_result = (
                        executed_target,
                        team_alignment,
                        self.day_count,
                        vote_log,
                    )
                else:
                    self._log(f"  - 투표 결과: 찬성 {self.final_vote}표 (과반 미달)")
                    self._log(
                        f"  - {executed_target}번 플레이어는 처형되지 않았습니다."
                    )
                    self.last_execution_result = None

        self._update_alive_status()

    def night_turn(self, ai_action: int):
        mafia_target = None
        doctor_target = None

        # 1. 마피아
        mafia_list = [
            p for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        ]
        if mafia_list:
            if (
                self.players[self.ai_id].role == config.ROLE_MAFIA
                and self.players[self.ai_id].alive
            ):
                mafia_target = ai_action
                self._log(
                    f"  - [마피아] AI({self.ai_id})가 {mafia_target}을(를) 지목했습니다."
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
            if doctor.id == self.ai_id:
                target = ai_action
                self._log(
                    f"  - [의사] AI({self.ai_id})이(가) {target}을(를) 치료했습니다."
                )
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
            if police.id == self.ai_id:
                target = ai_action
            else:
                target = police.decide_night_action(self.players, config.ROLE_POLICE)

            if target != -1:
                role_str = (
                    "마피아"
                    if self.players[target].role == config.ROLE_MAFIA
                    else "시민"
                )
                prefix = (
                    f"[경찰] AI({self.ai_id})"
                    if police.id == self.ai_id
                    else f"[경찰] {police.id}"
                )
                self._log(
                    f"  - {prefix}이(가) {target}을(를) 조사하여 '{role_str}'임을 확인했습니다."
                )

                if self.players[target].role == config.ROLE_MAFIA:
                    # 마피아로 확정
                    police.belief[target, 3] = 100  # 마피아 확신
                    police.belief[target, 0] = -100  # 시민 아님
                    police.confirmed_mafia.add(target)  # 확인된 마피아에 추가
                else:
                    # 시민으로 확정 (경찰, 의사, 시민 중 하나)
                    police.belief[target, 3] = -100  # 마피아 아님
                    police.belief[target, 0] = 50  # 시민일 가능성

        # 결과 정산
        no_death = False
        if mafia_target is not None:
            if mafia_target != doctor_target:
                self.players[mafia_target].alive = False
                self._log(f"  - {mafia_target}번 플레이어가 마피아에게 살해당했습니다.")
                no_death = False
            else:
                self._log(f"  - 의사가 {doctor_target}을(를) 살려냈습니다.")
                no_death = True

        # Store night result for belief updates (Doctor's Logic)
        self.last_night_result = {
            "no_death": no_death,
            "last_healed": doctor_target if doctor_target is not None else -1,
        }

        self._update_alive_status()

    # (이하 _update_alive_status, _get_game_status, check_game_over 메서드는 Agent 참조만 수정해서 사용)
    def _update_alive_status(self):
        self.alive_status = [1 if p.alive else 0 for p in self.players]

    def _get_game_status(self) -> Dict:
        phase_map = {
            config.PHASE_DAY_DISCUSSION: 0,  # Merged: claim + discussion
            config.PHASE_DAY_VOTE: 1,
            config.PHASE_NIGHT: 2,
        }
        return {
            "day": self.day_count,
            "phase": phase_map.get(self.phase, 0),
            "alive_status": self.alive_status,
            "roles": [p.role for p in self.players],
            "id": self.ai_id,
        }

    def check_game_over(self) -> Tuple[bool, bool]:
        mafia_count = sum(
            1 for p in self.players if p.role == config.ROLE_MAFIA and p.alive
        )
        citizen_count = sum(
            1 for p in self.players if p.role != config.ROLE_MAFIA and p.alive
        )

        # 무승부 조건: 최대 턴 수 초과
        if self.day_count > config.MAX_DAYS:
            self._log(f"\n게임 종료: {config.MAX_DAYS}일이 지나 무승부입니다!")
            return True, False  # 무승부는 패배로 처리

        if mafia_count == 0:
            self._log(f"\n게임 종료: 마피아가 모두 사망했습니다. 시민 팀 승리!")
            # [변경] 승패 판정 시 self.ai_id 사용
            return True, self.players[self.ai_id].role != config.ROLE_MAFIA
        elif mafia_count >= citizen_count:
            self._log(f"\n게임 종료: 마피아 승리!")
            # [변경]
            return True, self.players[self.ai_id].role == config.ROLE_MAFIA

        return False, False
