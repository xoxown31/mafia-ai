import sys
import os
import threading

from PyQt6.QtWidgets import QApplication
from core.managers.runner import train, test
from core.managers.experiment import ExperimentManager
from gui.launcher import Launcher

STOP = threading.Event()  # 종료 신호


def run_simulation(args):
    """
    AI 학습/테스트 로직
    ExperimentManager를 통해 설정 및 객체 생성 위임
    """
    if not getattr(args, "player_configs", None):
        print("Error: Player configurations not found. Please use the GUI.")
        return

    # ExperimentManager 초기화
    experiment = ExperimentManager(args)
    print(f"Simulation started: {experiment.logger.experiment_name}")

    try:
        # 환경 및 에이전트 생성
        env = experiment.build_env()
        agents = experiment.build_agents()
        rl_agents = experiment.get_rl_agents(agents)

        print(f"[{args.mode.upper()}] mode with {len(agents)} agents.")

        # 모드별 실행
        if args.mode == "train":
            env = experiment.build_vec_env(num_envs=8, num_cpus=4)
            train(
                env, 
                rl_agents, 
                agents, 
                args, 
                experiment.logger, 
                stop_event=STOP, 
            )

        elif args.mode == "test":
            test(env, agents, args, logger=experiment.logger, stop_event=STOP)

    finally:
        # 리소스 정리
        experiment.close()
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
