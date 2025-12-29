import matplotlib.pyplot as plt
import numpy as np
import os


def plot_results(rewards, win_rates, save_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "result.png")

    plt.figure(figsize=(12, 5))

    # 1. Total Reward
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Total Reward", alpha=0.6)
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
        plt.plot(
            range(len(rewards) - len(moving_avg), len(rewards)),
            moving_avg,
            color="red",
            label="Moving Avg (50)",
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.legend()
    plt.grid(True)

    # 2. Win Rate
    plt.subplot(1, 2, 2)
    plt.plot(win_rates, label="Win Rate (MA 100)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Trend")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # [추가] 저장할 폴더가 없으면 자동으로 생성하는 안전장치
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(save_path)
    print(f"\n[Info] 학습 결과 그래프가 '{save_path}'에 저장되었습니다.")
    plt.close()
