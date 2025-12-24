import argparse
import os
from core.env import MafiaEnv
from ai.ppo import PPO
from ai.reinforce import REINFORCEAgent
from core.runner import train, test  # 분리한 함수 임포트

def main():
    parser = argparse.ArgumentParser(description="Mafia AI Training/Testing Script")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'reinforce'], help='Agent type')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    args = parser.parse_args()

    # 로그 폴더 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "mafia_game_log.txt")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        # 환경 및 에이전트 초기화
        env = MafiaEnv(log_file=f)
        state_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.n

        if args.agent == 'ppo':
            agent = PPO(state_dim, action_dim)
        elif args.agent == 'reinforce':
            agent = REINFORCEAgent(state_dim, action_dim)
        
        # 모드별 실행 (runner.py로 위임)
        if args.mode == 'train':
            train(env, agent, args, f)
        elif args.mode == 'test':
            test(env, agent, args)

if __name__ == "__main__":
    main()