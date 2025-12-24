import argparse
import os
from core.env import MafiaEnv
from ai.ppo import PPO
from ai.reinforce import REINFORCEAgent
from core.runner import train, test
from utils.analysis import analyze_log_file

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
    
    # 1. 게임 실행 및 로그 기록
    with open(log_file_path, "w", encoding="utf-8") as f:
        # 환경 및 에이전트 초기화
        env = MafiaEnv(log_file=f)
        state_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.n

        if args.agent == 'ppo':
            agent = PPO(state_dim, action_dim)
        elif args.agent == 'reinforce':
            agent = REINFORCEAgent(state_dim, action_dim)
        
        # 모드별 실행
        if args.mode == 'train':
            train(env, agent, args, f)
        elif args.mode == 'test':
            test(env, agent, args)

    # 2. [추가] 학습 종료 후 로그 정밀 분석 실행
    # 파일을 닫은(with문 종료) 후에 실행해야 안전하게 읽을 수 있습니다.
    if args.mode == 'train':
        print("\n" + "="*30)
        print(" Start Post-Training Analysis")
        print("="*30)
        analyze_log_file(log_file_path, output_img="analysis_detailed.png")

if __name__ == "__main__":
    main()