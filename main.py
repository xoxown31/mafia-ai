import argparse
from core.env import MafiaEnv
from ai.ppo import PPO # (나중에 구현 예정)

def main(args):
    # 1. 환경 생성
    env = MafiaEnv()
    print(f"Environment Initialized. Action Space: {env.action_space}")

    # 2. 모드에 따른 실행
    if args.mode == 'train':
        print("Training Mode Started...")
        # TODO: 학습 루프 구현
        pass
        
    elif args.mode == 'test':
        print("Test Mode Started...")
        obs, _ = env.reset()
        done = False
        while not done:
            # 임시 랜덤 행동
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            env.render()
        print("Game Over")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='train or test')
    args = parser.parse_args()
    main(args)