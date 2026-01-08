import gymnasium as gym
import supersuit as ss
import os

from typing import Dict, Any, List
from pathlib import Path


# [중요] 클래스 밖으로 뺀 함수
# 이 함수는 "로그 파일(logger)" 없이 순수한 게임 환경만 만듭니다.
# 워커 프로세스들은 이 함수를 통해 만들어지므로, 파일 충돌이 안 납니다.
def make_env_for_worker():
    from core.envs.mafia_env import MafiaEnv

    # 워커는 로그를 파일에 안 씀 (logger=None)
    return MafiaEnv(logger=None)


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
                if p_config["type"] == "rl":
                    experiment_name = (
                        f"{p_config['algo']}_{p_config['backbone']}_{self.mode}"
                    )
                    break

        log_dir = str(getattr(self.args, "paths", {}).get("log_dir", "logs"))
        # 메인 프로세스는 정상적으로 파일을 씀
        return LogManager(
            experiment_name=experiment_name, log_dir=log_dir, write_mode=True
        )

    def build_env(self) -> MafiaEnv:
        """메인 프로세스용 환경 (여기엔 로거가 있음)"""
        return MafiaEnv(logger=self.logger)

    def build_vec_env(self, num_envs: int = 8, num_cpus: int = 4):
        """
        병렬 학습 환경 생성
        """
        print(f"[System] Building Parallel Env: {num_envs} games with {num_cpus} CPUs")

        # 1. 아까 밖으로 빼둔 함수(make_env_for_worker)를 사용해 템플릿을 만듭니다.
        # 이렇게 하면 'LogManager'가 묻지 않은 깨끗한 환경이 만들어집니다.
        env = make_env_for_worker()

        # 2. PettingZoo -> Gymnasium 변환
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        # 3. 병렬 연결 (이제 에러가 안 날 겁니다)
        try:
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="gymnasium"
            )
        except Exception as e:
            # 혹시라도 또 에러가 나면 안전하게 싱글 프로세스로 전환
            print(f"[Error] Parallel creation failed: {e}")
            print("[System] Switching to single process mode (Safe Mode)")
            vec_env = ss.concat_vec_envs_v1(
                env, num_vec_envs=num_envs, num_cpus=0, base_class="gymnasium"
            )

        return vec_env

    def build_agents(self) -> Dict[int, Any]:
        state_dim = config.game.OBS_DIM
        agents = {}

        if not self.player_configs:
            # Fallback or error? main.py raised error.
            return {}

        for i, p_config in enumerate(self.player_configs):
            if p_config["type"] == "rl":
                agent = RLAgent(
                    player_id=i,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dims=config.game.ACTION_DIMS,
                    algorithm=p_config["algo"],
                    backbone=p_config["backbone"],
                    hidden_dim=p_config.get("hidden_dim", 128),
                    num_layers=p_config.get("num_layers", 2),
                )

                load_path = p_config.get("load_model_path")

                # 모델 경로
                if load_path:
                    if os.path.exists(load_path):
                        try:
                            agent.load(load_path)  # RLAgent의 load 메서드 호출
                            print(
                                f"[Experiment] Agent {i}: 모델 로드 성공 ({load_path})"
                            )
                        except Exception as e:
                            print(f"[Experiment] Agent {i}: 모델 로드 실패! ({e})")
                    else:
                        print(
                            f"[Experiment] Agent {i}: 경로에 파일이 없습니다 ({load_path})"
                        )

                agents[i] = agent
            elif p_config["type"] == "llm":
                # ... (LLM 에이전트 생성 코드는 그대로 둠)
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
