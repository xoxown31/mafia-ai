import sys
import argparse
import os
import threading
import tkinter as tk

from core.env import MafiaEnv
from core.game import MafiaGame
from core.agent.rlAgent import RLAgent
from config import Role
from PyQt6.QtWidgets import QApplication
from core.runner import train, test
from utils.analysis import analyze_log_file
from gui.launcher import Launcher

# GUI 모듈 임포트 (gui 패키지가 없어도 에러 안 나게 처리)
try:
    from gui.gui_viewer import MafiaLogViewerApp

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


def run_simulation(args):
    """
    기존 main() 함수에 있던 AI 학습/테스트 로직을 별도 함수로 분리
    """

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "mafia_game_log.txt")

    print(f"Simulation started with {args.agent} agent.")

    # 1. 게임 실행 및 로그 기록
    with open(log_file_path, "w", encoding="utf-8") as f:
        # LLM 에이전트 모드
        if args.agent == "llm":
            print("Running LLM-only simulation.")
            game = MafiaGame(log_file=f)
            game.reset()
            done = False
            while not done:
                # LLM 에이전트는 내부적으로 스스로 행동을 결정하므로,
                # process_turn에 전달하는 action은 의미 없음
                status, done, win = game.process_turn()
            print("\nLLM simulation finished. Winner determined by game log.")

        # PPO 또는 REINFORCE 에이전트 모드
        else:
            print(f"[{args.mode.upper()}] mode for {args.agent.upper()} agent.")
            # 환경 및 에이전트 초기화
            env = MafiaEnv(log_file=f)
            state_dim = env.observation_space["observation"].shape[0]
            action_dim = env.action_space.n

            # RLAgent 인스턴스 생성
            agent = RLAgent(
                player_id=0,
                role=Role.CITIZEN,
                state_dim=state_dim,
                action_dim=action_dim,
                algorithm=args.agent,
                backbone=getattr(args, 'backbone', 'mlp'),
                use_il=getattr(args, 'use_il', False),
                hidden_dim=getattr(args, 'hidden_dim', 128),
                num_layers=getattr(args, 'num_layers', 2)
            )

            # 모드별 실행
            if args.mode == "train":
                train(env, agent, args, f)
            elif args.mode == "test":
                test(env, agent, args)

    # 2. 학습 종료 후 로그 정밀 분석 실행 (train 모드였을 경우)
    if args.agent != "llm" and args.mode == "train":
        print("\n" + "=" * 30)
        print(" Start Post-Training Analysis")
        print("=" * 30)
        try:
            analyze_log_file(log_file_path, output_img="analysis_detailed.png")
        except Exception as e:
            print(f"Analysis failed: {e}")

    print("Simulation finished.")


def start_gui():
    app = QApplication(sys.argv)
    launcher = Launcher()

    def on_simulation_start(args):
        sim_thread = threading.Thread(target=run_simulation, args=(args,), daemon=True)
        sim_thread.start()

    # 시그널 연결
    launcher.start_simulation_signal.connect(on_simulation_start)

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
            "--backbone", type=str, default="mlp", choices=["mlp", "lstm", "gru"],
            help="Neural network backbone"
        )
        parser.add_argument(
            "--use_il", action="store_true",
            help="Enable Imitation Learning"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=128,
            help="Hidden dimension for neural network"
        )
        parser.add_argument(
            "--num_layers", type=int, default=2,
            help="Number of layers for RNN"
        )
        
        args = parser.parse_args()

        if args.gui and GUI_AVAILABLE:
            # 기존 Tkinter 뷰어 실행 로직 (유지)
            print("Launching Legacy GUI with Simulation...")
            sim_thread = threading.Thread(
                target=run_simulation, args=(args,), daemon=True
            )
            sim_thread.start()
            root = tk.Tk()
            root.mainloop()
        else:
            run_simulation(args)

    # 인자가 없으면 -> GUI 실행
    else:
        print("Start GUI")
        start_gui()


if __name__ == "__main__":
    main()
