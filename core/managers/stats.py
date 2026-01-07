from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict
from config import Role, EventType, Phase

class StatsManager:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_wins = defaultdict(list)
        self.recent_mafia_wins = []
        self.recent_citizen_wins = []
        self.mafia_win_days = []
        self.citizen_win_days = []

    def calculate_stats(
        self, 
        env, 
        rl_agents: Dict[int, Any], 
        all_agents: Dict[int, Any],
        episode_rewards: Dict[int, float], 
        is_wins: Dict[int, bool],
        train_metrics: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, float]:
        
        metrics = {}
        game = env.game
        
        # --- 1. Agent Stats (Brain) ---
        mafia_rewards = []
        citizen_rewards = []

        for pid in rl_agents.keys():
            # Update recent wins
            win = 1 if is_wins.get(pid, False) else 0
            self.recent_wins[pid].append(win)
            if len(self.recent_wins[pid]) > self.window_size:
                self.recent_wins[pid].pop(0)
            
            win_rate = np.mean(self.recent_wins[pid]) if self.recent_wins[pid] else 0.0
            
            reward = episode_rewards.get(pid, 0.0)
            metrics[f"Agent_{pid}/Reward_Total"] = reward
            metrics[f"Agent_{pid}/Win_Rate"] = win_rate

            # Collect Team Rewards
            if all_agents[pid].role == Role.MAFIA:
                mafia_rewards.append(reward)
            else:
                citizen_rewards.append(reward)

        metrics["Reward/Total"] = sum(episode_rewards.values())
        metrics["Reward/Mafia_Avg"] = np.mean(mafia_rewards) if mafia_rewards else 0.0
        metrics["Reward/Citizen_Avg"] = np.mean(citizen_rewards) if citizen_rewards else 0.0

        # Team/Role Training Stats
        for role_key in ["Mafia", "Citizen"]:
            role_metrics = train_metrics.get(role_key, {})
            if "loss" in role_metrics and role_metrics["loss"]:
                metrics[f"Train/{role_key}_Loss"] = np.mean(role_metrics["loss"])
            if "entropy" in role_metrics and role_metrics["entropy"]:
                metrics[f"Train/{role_key}_Entropy"] = np.mean(role_metrics["entropy"])
            if "approx_kl" in role_metrics and role_metrics["approx_kl"]:
                metrics[f"Train/{role_key}_ApproxKL"] = np.mean(role_metrics["approx_kl"])
            if "clip_frac" in role_metrics and role_metrics["clip_frac"]:
                metrics[f"Train/{role_key}_ClipFrac"] = np.mean(role_metrics["clip_frac"])

        # --- 2. Game Stats (Behavior) ---
        metrics["Game/Duration"] = game.day
        
        # Determine Game Winner Team
        mafia_won = False
        citizen_won = False
        
        last_event = game.history[-1] if game.history else None
        if last_event and last_event.phase == Phase.GAME_END:
            citizen_won_game = last_event.value
            if citizen_won_game:
                citizen_won = True
            else:
                mafia_won = True
        else:
            # Fallback: check RL agents wins
            for pid, won in is_wins.items():
                if won:
                    if all_agents[pid].role == Role.MAFIA:
                        mafia_won = True
                    else:
                        citizen_won = True
                    break 
        
        self.recent_mafia_wins.append(1 if mafia_won else 0)
        self.recent_citizen_wins.append(1 if citizen_won else 0)
        
        if mafia_won:
            self.mafia_win_days.append(game.day)
        if citizen_won:
            self.citizen_win_days.append(game.day)

        if len(self.recent_mafia_wins) > self.window_size: self.recent_mafia_wins.pop(0)
        if len(self.recent_citizen_wins) > self.window_size: self.recent_citizen_wins.pop(0)
        if len(self.mafia_win_days) > self.window_size: self.mafia_win_days.pop(0)
        if len(self.citizen_win_days) > self.window_size: self.citizen_win_days.pop(0)
        
        metrics["Game/Mafia_WinRate"] = np.mean(self.recent_mafia_wins) if self.recent_mafia_wins else 0.0
        metrics["Game/Citizen_WinRate"] = np.mean(self.recent_citizen_wins) if self.recent_citizen_wins else 0.0
        metrics["Game/Avg_Day_When_Mafia_Wins"] = np.mean(self.mafia_win_days) if self.mafia_win_days else 0.0
        metrics["Game/Avg_Day_When_Citizen_Wins"] = np.mean(self.citizen_win_days) if self.citizen_win_days else 0.0

        # Action Stats
        action_metrics = self._analyze_history(game)
        metrics.update(action_metrics)
        
        return metrics

    def _analyze_history(self, game) -> Dict[str, float]:
        metrics = {}
        
        # Action Counter
        mafia_kill_attempts = 0
        mafia_kill_success = 0
        doctor_save_success = 0
        doctor_self_heal = 0
        doctor_total_protects = 0   # 의사 총 활동 횟수 (분모용)
        
        police_investigations = 0
        police_finds = 0
        
        # Vote Counter
        vote_total = 0
        vote_abstain = 0
        mafia_betrayal = 0
        citizen_correct_vote = 0
        
        mafia_votes = 0      # 마피아 총 투표 (분모)
        citizen_votes = 0    # 시민 팀 총 투표 (분모)
        
        # Execution Counter
        execution_total = 0
        mafia_executed = 0
        citizen_sacrificed = 0
        
        # 1. 밤(Night) 상호작용 분석 (Kill vs Protect)
        night_events = [e for e in game.history if e.phase == Phase.NIGHT]
        
        for d in range(1, game.day + 1):
            day_night_events = [e for e in night_events if e.day == d]
            kill_event = next((e for e in day_night_events if e.event_type == EventType.KILL), None)
            protect_event = next((e for e in day_night_events if e.event_type == EventType.PROTECT), None)
            
            if protect_event:
                doctor_total_protects += 1 # 의사 활동 카운트
                if protect_event.actor_id == protect_event.target_id:
                    doctor_self_heal += 1

            if kill_event:
                mafia_kill_attempts += 1
                is_saved = False
                
                if protect_event and kill_event.target_id == protect_event.target_id:
                    is_saved = True
                    doctor_save_success += 1
                
                if not is_saved:
                    mafia_kill_success += 1
                    
        # 2. 전체 이벤트 루프 (한 번만 순회하도록 통합)
        for event in game.history:
            # Police
            if event.event_type == EventType.POLICE_RESULT:
                police_investigations += 1
                if event.value == Role.MAFIA:
                    police_finds += 1
            
            # Vote
            elif event.event_type == EventType.VOTE:
                vote_total += 1
                actor = game.players[event.actor_id]
                
                # 분모(총 투표 수) 카운트
                if actor.role == Role.MAFIA:
                    mafia_votes += 1
                else:
                    citizen_votes += 1

                if event.target_id == -1:
                    vote_abstain += 1
                else:
                    target = game.players[event.target_id]
                    # 분자(배신/정답) 카운트
                    if actor.role == Role.MAFIA and target.role == Role.MAFIA:
                        mafia_betrayal += 1
                    
                    if actor.role != Role.MAFIA and target.role == Role.MAFIA:
                        citizen_correct_vote += 1
            
            # Execute
            elif event.event_type == EventType.EXECUTE:
                if event.target_id != -1:
                    execution_total += 1
                    target = game.players[event.target_id]
                    if target.role == Role.MAFIA:
                        mafia_executed += 1
                    else:
                        citizen_sacrificed += 1
                        
        # Rate Calculation
        metrics["Vote/Abstain_Rate"] = vote_abstain / vote_total if vote_total > 0 else 0.0
        metrics["Vote/Mafia_Betrayal_Rate"] = mafia_betrayal / mafia_votes if mafia_votes > 0 else 0.0
        metrics["Vote/Citizen_Accuracy_Rate"] = citizen_correct_vote / citizen_votes if citizen_votes > 0 else 0.0
        
        metrics["Action/Doctor_Save_Rate"] = doctor_save_success / mafia_kill_attempts if mafia_kill_attempts > 0 else 0.0
        # 의사 자가 치료율 분모를 '의사 총 활동 수'로 사용
        metrics["Action/Doctor_Self_Heal_Rate"] = doctor_self_heal / doctor_total_protects if doctor_total_protects > 0 else 0.0
        
        metrics["Action/Police_Find_Rate"] = police_finds / police_investigations if police_investigations > 0 else 0.0
        metrics["Action/Mafia_Kill_Success_Rate"] = mafia_kill_success / mafia_kill_attempts if mafia_kill_attempts > 0 else 0.0
        
        # 처형 빈도 (Executions per day)
        metrics["Game/Execution_Frequency"] = execution_total / game.day if game.day > 0 else 0.0
        
        metrics["Vote/Mafia_Lynch_Rate"] = mafia_executed / execution_total if execution_total > 0 else 0.0
        metrics["Vote/Citizen_Sacrifice_Rate"] = citizen_sacrificed / execution_total if execution_total > 0 else 0.0

        return metrics
