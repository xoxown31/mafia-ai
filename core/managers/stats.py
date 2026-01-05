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
        for pid in rl_agents.keys():
            # Update recent wins
            win = 1 if is_wins.get(pid, False) else 0
            self.recent_wins[pid].append(win)
            if len(self.recent_wins[pid]) > self.window_size:
                self.recent_wins[pid].pop(0)
            
            win_rate = np.mean(self.recent_wins[pid]) if self.recent_wins[pid] else 0.0
            
            metrics[f"Agent_{pid}/Reward_Total"] = episode_rewards.get(pid, 0.0)
            metrics[f"Agent_{pid}/Win_Rate"] = win_rate

        # Team/Role Training Stats
        for role_key in ["Mafia", "Citizen"]:
            if train_metrics[role_key]["loss"]:
                metrics[f"Train/{role_key}_Loss"] = np.mean(train_metrics[role_key]["loss"])
            if train_metrics[role_key]["entropy"]:
                metrics[f"Train/{role_key}_Entropy"] = np.mean(train_metrics[role_key]["entropy"])

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
        
        if len(self.recent_mafia_wins) > self.window_size: self.recent_mafia_wins.pop(0)
        if len(self.recent_citizen_wins) > self.window_size: self.recent_citizen_wins.pop(0)
        
        metrics["Game/Mafia_WinRate"] = np.mean(self.recent_mafia_wins) if self.recent_mafia_wins else 0.0
        metrics["Game/Citizen_WinRate"] = np.mean(self.recent_citizen_wins) if self.recent_citizen_wins else 0.0

        # Action Stats
        action_metrics = self._analyze_history(game)
        metrics.update(action_metrics)
        
        return metrics

    def _analyze_history(self, game) -> Dict[str, float]:
        metrics = {}
        mafia_kill_attempts = 0
        doctor_saves = 0
        police_investigations = 0
        police_finds = 0
        lynch_executions = 0
        mafia_lynched = 0
        citizen_lynched = 0
        
        night_events = [e for e in game.history if e.phase == Phase.NIGHT]
        for d in range(1, game.day + 1):
            day_events = [e for e in night_events if e.day == d]
            kill_event = next((e for e in day_events if e.event_type == EventType.KILL), None)
            protect_event = next((e for e in day_events if e.event_type == EventType.PROTECT), None)
            
            if kill_event:
                mafia_kill_attempts += 1
                if protect_event and kill_event.target_id == protect_event.target_id:
                    doctor_saves += 1
            
            police_events = [e for e in day_events if e.event_type == EventType.POLICE_RESULT]
            for pe in police_events:
                police_investigations += 1
                if pe.value == Role.MAFIA:
                    police_finds += 1
        
        execute_events = [e for e in game.history if e.event_type == EventType.EXECUTE and e.target_id != -1]
        for ee in execute_events:
            lynch_executions += 1
            target = game.players[ee.target_id]
            if target.role == Role.MAFIA:
                mafia_lynched += 1
            else:
                citizen_lynched += 1
        
        metrics["Action/Doctor_Save_Rate"] = doctor_saves / mafia_kill_attempts if mafia_kill_attempts > 0 else 0.0
        metrics["Action/Police_Find_Rate"] = police_finds / police_investigations if police_investigations > 0 else 0.0
        metrics["Vote/Mafia_Lynch_Rate"] = mafia_lynched / lynch_executions if lynch_executions > 0 else 0.0
        metrics["Vote/Wrong_Lynch_Rate"] = citizen_lynched / lynch_executions if lynch_executions > 0 else 0.0
        
        return metrics
