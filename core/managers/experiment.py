from typing import Dict, Any, List
from pathlib import Path
from core.envs.mafia_env import MafiaEnv
from core.agents.rl_agent import RLAgent
from core.agents.llm_agent import LLMAgent
from core.managers.logger import LogManager
from config import Role, config

class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.player_configs = getattr(args, "player_configs", [])
        self.mode = args.mode
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> LogManager:
        experiment_name = f"llm_{self.mode}"
        if self.player_configs:
            for p_config in self.player_configs:
                if p_config['type'] == 'rl':
                    experiment_name = f"{p_config['algo']}_{p_config['backbone']}_{self.mode}"
                    break
        
        log_dir = str(getattr(self.args, "paths", {}).get("log_dir", "logs"))
        return LogManager(experiment_name=experiment_name, log_dir=log_dir)

    def build_env(self) -> MafiaEnv:
        return MafiaEnv(logger=self.logger)

    def build_agents(self) -> Dict[int, Any]:
        state_dim = config.game.OBS_DIM
        agents = {}
        
        if not self.player_configs:
             # Fallback or error? main.py raised error.
             return {}

        for i, p_config in enumerate(self.player_configs):
            if p_config['type'] == 'rl':
                agent = RLAgent(
                    player_id=i,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dims=config.game.ACTION_DIMS,
                    algorithm=p_config['algo'],
                    backbone=p_config['backbone'],
                    hidden_dim=p_config.get('hidden_dim', 128),
                    num_layers=p_config.get('num_layers', 2),
                )
                agents[i] = agent
            elif p_config['type'] == 'llm':
                agent = LLMAgent(player_id=i, logger=self.logger)
                agents[i] = agent
            else:
                pass
                
        return agents

    def get_rl_agents(self, agents: Dict[int, Any]) -> Dict[int, Any]:
        return {i: a for i, a in agents.items() if isinstance(a, RLAgent)}

    def close(self):
        if self.logger:
            self.logger.close()
