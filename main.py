import argparse
import os
import threading
import tkinter as tk

from core.env import MafiaEnv
from ai.ppo import PPO
from ai.reinforce import REINFORCEAgent
from utils.runner import train, test
from utils.analysis import analyze_log_file

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
    # 로그 폴더 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "mafia_game_log.txt")

    print(f"[{args.mode.upper()}] Simulation started with {args.agent} agent.")

    # 1. 게임 실행 및 로그 기록
    with open(log_file_path, "w", encoding="utf-8") as f:
        # 환경 및 에이전트 초기화
        env = MafiaEnv(log_file=f)
        state_dim = env.observation_space["observation"].shape[0]
        action_dim = env.action_space.n

        if args.agent == "ppo":
            agent = PPO(state_dim, action_dim)
        elif args.agent == "reinforce":
            agent = REINFORCEAgent(state_dim, action_dim)

        # 모드별 실행
        if args.mode == "train":
            train(env, agent, args, f)
        elif args.mode == "test":
            test(env, agent, args)

    # 2. 학습 종료 후 로그 정밀 분석 실행
    if args.mode == "train":
        print("\n" + "=" * 30)
        print(" Start Post-Training Analysis")
        print("=" * 30)
        # analyze_log_file 함수가 메인 스레드 UI와 충돌하지 않도록 주의
        try:
            analyze_log_file(log_file_path, output_img="analysis_detailed.png")
        except Exception as e:
            print(f"Analysis failed: {e}")

    print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description="Mafia AI Training/Testing Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Execution mode",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "reinforce"],
        help="Agent type",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch GUI viewer alongside simulation"
    )
    args = parser.parse_args()

    # GUI 실행 옵션이 켜져 있고, GUI 모듈을 불러올 수 있다면
    if args.gui and GUI_AVAILABLE:
        print("Launching GUI with Simulation...")

        # 1. AI 시뮬레이션을 별도 스레드로 실행 (Daemon=True: 창 끄면 같이 꺼짐)
        sim_thread = threading.Thread(target=run_simulation, args=(args,), daemon=True)
        sim_thread.start()

        # 2. 메인 스레드에서 GUI 실행
        root = tk.Tk()
        app = MafiaLogViewerApp(root)

        root.mainloop()

    else:
        # GUI 없이 실행 (기존 방식)
        if args.gui and not GUI_AVAILABLE:
            print("Warning: GUI module not found. Running in console mode.")
        run_simulation(args)


if __name__ == "__main__":
    main()
