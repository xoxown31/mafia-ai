import sys
import os
import threading

from core.env import MafiaEnv
from core.game import MafiaGame
from core.agent.rlAgent import RLAgent
from core.agent.llmAgent import LLMAgent
from core.logger import LogManager
from config import Role, config
from PyQt6.QtWidgets import QApplication
from core.runner import train, test

from gui.launcher import Launcher

STOP = threading.Event()  # 종료 신호


def run_simulation(args):
    """
    AI 학습/테스트 로직
    LogManager를 통한 통합 로깅 시스템 사용
    GUI에서 전달된 player_configs 구조 사용
    """
    player_configs = getattr(args, "player_configs", None)

    if player_configs is None:
        print("Error: Player configurations not found. Please use the GUI.")
        return

    # Player 0의 설정으로 실험 이름 생성 -> 첫 번째 RL 에이전트 기준
    experiment_name = f"llm_{args.mode}"  # 기본값
    for config in player_configs:
        if config["type"] == "rl":
            experiment_name = f"{config['algo']}_{config['backbone']}_{args.mode}"
            break

    # LogManager 초기화
    log_dir = str(getattr(args, "paths", {}).get("log_dir", "logs"))
    logger = LogManager(experiment_name=experiment_name, log_dir=log_dir)

    print(f"Simulation started: {experiment_name}")
    print(
        f"Player configurations: {player_configs if player_configs else 'Legacy mode'}"
    )

    try:
        if player_configs is None:
            raise ValueError(
                "Player configurations must be provided. CLI mode is deprecated."
            )

        print(f"[{args.mode.upper()}] mode with player_configs.")

        # 환경 초기화 (PettingZoo)
        env = MafiaEnv(logger=logger)
        agent_id = env.possible_agents[0]
        # state_dim = env.observation_space(agent_id)["observation"].shape[0]
        # 78차원으로 고정
        state_dim = 78

        # 모든 에이전트 생성 (RL 및 LLM)
        agents = {}
        for i, config in enumerate(player_configs):
            if config["type"] == "rl":
                agent = RLAgent(
                    player_id=i,
                    role=Role.CITIZEN,  # 역할은 게임 내에서 동적으로 할당됨
                    state_dim=state_dim,
                    action_dims=[9, 5],
                    algorithm=config["algo"],
                    backbone=config["backbone"],
                    use_il=False,
                    hidden_dim=config.get("hidden_dim", 128),
                    num_layers=config.get("num_layers", 2),
                )
                agents[i] = agent
            elif config["type"] == "llm":
                agent = LLMAgent(player_id=i, logger=logger)
                agents[i] = agent
            else:
                # 기본 봇 또는 다른 타입 (필요시 추가)
                pass

        # 모드별 실행
        if args.mode == "train":
            # 학습 모드에서는 RL 에이전트만 필터링하여 전달 (LLM은 runner에서 자동 처리)
            # 하지만 runner가 모든 에이전트를 관리하도록 변경했으므로 전체 전달
            # 단, train 함수는 RL 에이전트의 update를 호출하므로 RL 에이전트 식별이 필요함
            # runner.train 내부에서 isinstance 체크 또는 별도 분리 가능
            # 여기서는 RL 에이전트만 추출하여 전달하고, 나머지는 runner가 env에서 찾지 않고
            # agents 딕셔너리에서 찾도록 runner를 수정해야 함.

            # 현재 runner.train은 agents 딕셔너리를 받아서 RL 에이전트로 취급하고 있음.
            # 따라서 RL 에이전트만 담긴 딕셔너리와, 전체 에이전트 리스트를 분리해서 넘기거나
            # runner가 타입을 체크하도록 해야 함.

            # runner.train의 시그니처를 변경하여 (env, rl_agents, all_agents, ...) 형태로 하거나
            # agents 딕셔너리에 모두 넣고 runner 내부에서 구분.

            # 여기서는 RL 에이전트만 추출해서 넘기고, LLM 에이전트는 runner가 env.game.players 대신
            # 별도로 전달받은 all_agents 딕셔너리를 사용하도록 runner를 수정하는 것이 좋음.

            rl_agents = {i: a for i, a in agents.items() if isinstance(a, RLAgent)}
            train(env, rl_agents, agents, args, logger, stop_event=STOP)

        elif args.mode == "test":
            test(env, agents, args, stop_event=STOP)

    finally:
        # LogManager 리소스 정리
        logger.close()
        print("Simulation finished.")


def start_gui():
    app = QApplication(sys.argv)
    launcher = Launcher()

    def on_simulation_start(args):
        STOP.clear()
        sim_thread = threading.Thread(target=run_simulation, args=(args,), daemon=True)
        sim_thread.start()

    def on_simulation_stop():
        print("시뮬레이션 종료")
        STOP.set()

    # 시그널 연결
    launcher.start_simulation_signal.connect(on_simulation_start)
    launcher.stop_simulation_signal.connect(on_simulation_stop)

    launcher.show()
    sys.exit(app.exec())


def main():
    print("Starting Mafia AI GUI...")
    start_gui()


if __name__ == "__main__":
    main()
