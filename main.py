import sys
import argparse
import os
import threading

from core.env import MafiaEnv
from core.game import MafiaGame
from core.agent.rlAgent import RLAgent
from core.agent.llmAgent import LLMAgent
from core.logger import LogManager
from config import Role
from PyQt6.QtWidgets import QApplication
from core.runner import train, test

from gui.launcher import Launcher

STOP = threading.Event()  # 종료 신호


def _run_full_game_simulation(player_configs, args, logger):
    """
    player_configs에 따라 LLM/RL 혼합 또는 순수 LLM 게임을 실행
    """
    from core.agent.baseAgent import BaseAgent

    # 각 플레이어 설정에 맞는 에이전트 생성
    agents = []
    for i, config in enumerate(player_configs):
        if config["type"] == "llm":
            agent = LLMAgent(player_id=i, logger=logger)
        else:  # rl
            # RL 에이전트는 현재 MafiaGame과 호환되지 않을 수 있음
            # 추후 BaseAgent 기반으로 통합 필요
            print(
                f"Warning: RL agent in position {i} not fully supported in game simulation yet"
            )
            # 임시로 LLM으로 대체
            agent = LLMAgent(player_id=i, logger=logger)
        agents.append(agent)

    # MafiaGame 생성 및 실행
    game = MafiaGame(agents=agents, logger=logger)

    episodes = getattr(args, "episodes", 1)
    print(f"Running {episodes} game(s) with player configuration:")
    for i, config in enumerate(player_configs):
        print(f"  Player {i}: {config['type'].upper()}")

    # 에피소드 실행
    for ep in range(episodes):
        if STOP.is_set():
            print("\n[System] Simulation stopped by user (Episode Loop).")
            break

        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"{'='*50}")

        status = game.reset(agents=agents)

        # 역할 출력
        print("\n[Role Assignment]")
        for p in game.players:
            print(f"  Player {p.id}: {p.role}")

        turn = 0
        max_turns = 50  # 무한 루프 방지

        while turn < max_turns:
            if STOP.is_set():
                print(f"\n[System] Simulation stopped by user at Turn {turn}.")
                break

            status, is_over, is_win = game.process_turn()
            turn += 1

            if is_over:
                result = "CITIZEN WIN" if is_win else "MAFIA WIN"
                print(f"\n{'='*50}")
                print(f"Game Over! {result}")
                print(f"Total turns: {turn}")
                print(f"{'='*50}\n")
                break
        else:
            print(f"\nGame reached maximum turn limit ({max_turns})")

    print(f"\nCompleted {episodes} episode(s)")


def run_simulation(args):
    """
    AI 학습/테스트 로직
    LogManager를 통한 통합 로깅 시스템 사용
    새로운 player_configs 구조 지원
    """
    # 하위 호환성: 이전 구조(args.agent) 지원
    if hasattr(args, "agent"):
        # CLI에서 실행된 경우 (레거시 구조)
        player_configs = None
        experiment_name = f"{args.agent}_{getattr(args, 'backbone', 'mlp')}_{args.mode}"
    else:
        # GUI에서 실행된 경우 (새로운 구조)
        player_configs = args.player_configs

        # Player 0의 설정으로 실험 이름 생성
        player0_config = player_configs[0]
        if player0_config["type"] == "rl":
            experiment_name = (
                f"{player0_config['algo']}_{player0_config['backbone']}_{args.mode}"
            )
        else:
            experiment_name = f"llm_{args.mode}"

    # LogManager 초기화
    log_dir = str(getattr(args, "paths", {}).get("log_dir", "logs"))
    logger = LogManager(experiment_name=experiment_name, log_dir=log_dir)

    print(f"Simulation started: {experiment_name}")
    print(
        f"Player configurations: {player_configs if player_configs else 'Legacy mode'}"
    )

    try:
        # 레거시 구조 처리
        if player_configs is None:
            # LLM 에이전트 모드
            if args.agent == "llm":
                print("Running LLM-only simulation.")
                # TODO: LLM 전용 시뮬레이션 로직 구현
                print("LLM simulation finished.")

            # RL 에이전트 모드 (PPO, REINFORCE)
            else:
                print(f"[{args.mode.upper()}] mode for {args.agent.upper()} agent.")

                # 환경 및 에이전트 초기화
                env = MafiaEnv()
                state_dim = env.observation_space["observation"].shape[0]
                action_dim = env.action_space.n

                agent = RLAgent(
                    player_id=0,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    algorithm=args.agent,
                    backbone=getattr(args, "backbone", "mlp"),
                    use_il=getattr(args, "use_il", False),
                    hidden_dim=getattr(args, "hidden_dim", 128),
                    num_layers=getattr(args, "num_layers", 2),
                )

                # 모드별 실행
                if args.mode == "train":
                    train(env, agent, args, logger)
                elif args.mode == "test":
                    test(env, agent, args)

        # 새로운 player_configs 구조 처리
        else:
            print(f"[{args.mode.upper()}] mode with player_configs.")

            # Player 0 설정 추출
            player0_config = player_configs[0]

            # Player 0이 RL인 경우 학습/테스트 진행
            if player0_config["type"] == "rl":
                env = MafiaEnv()
                state_dim = env.observation_space["observation"].shape[0]
                action_dim = env.action_space.n

                agent = RLAgent(
                    player_id=0,
                    role=Role.CITIZEN,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    algorithm=player0_config["algo"],
                    backbone=player0_config["backbone"],
                    use_il=False,  # GUI에서는 기본적으로 IL 비활성화
                    hidden_dim=player0_config.get("hidden_dim", 128),
                    num_layers=player0_config.get("num_layers", 2),
                )

                # 모드별 실행
                if args.mode == "train":
                    train(env, agent, args, logger)
                elif args.mode == "test":
                    test(env, agent, args)

            # Player 0이 LLM인 경우 또는 혼합 구성인 경우 - 풀 게임 시뮬레이션
            else:
                print("Running full game simulation with mixed/LLM agents.")
                _run_full_game_simulation(player_configs, args, logger)

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
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Mafia AI Training/Testing Script")
        parser.add_argument(
            "--mode", type=str, default="train", choices=["train", "test"]
        )
        parser.add_argument(
            "--agent", type=str, default="ppo", choices=["ppo", "reinforce", "llm"]
        )
        parser.add_argument("--episodes", type=int, default=1000)
        parser.add_argument("--gui", action="store_true")

        # RLAgent 설정
        parser.add_argument(
            "--backbone",
            type=str,
            default="mlp",
            choices=["mlp", "lstm", "gru"],
            help="Neural network backbone",
        )
        parser.add_argument(
            "--use_il", action="store_true", help="Enable Imitation Learning"
        )
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="Hidden dimension for neural network",
        )
        parser.add_argument(
            "--num_layers", type=int, default=2, help="Number of layers for RNN"
        )

        args = parser.parse_args()
        run_simulation(args)

    # 인자가 없으면 -> GUI 실행
    else:
        print("Start GUI")
        start_gui()


if __name__ == "__main__":
    main()
