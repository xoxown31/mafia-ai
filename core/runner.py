from utils.visualize import plot_results


def train(env, agent, args, log_file):
    """학습 모드 실행"""
    algorithm_name = getattr(agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(agent, 'backbone', 'mlp').upper()
    print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")

    history_rewards = []
    history_win_rates = []
    recent_wins = []

    for episode in range(1, args.episodes + 1):
        log_file.write(f"\n{'='*20} Episode {episode} Start {'='*20}\n")
        
        # RNN 은닉 상태 초기화
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        is_win = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            # RLAgent의 인터페이스 사용
            agent.store_reward(reward, done)

            # 승리 판정
            if done and reward > 5.0:
                is_win = True

            obs = next_obs
            total_reward += reward

        # 에피소드 종료 후 학습
        agent.update()

        # 지표 기록
        history_rewards.append(total_reward)
        recent_wins.append(1 if is_win else 0)
        if len(recent_wins) > 100:
            recent_wins.pop(0)

        current_win_rate = sum(recent_wins) / len(recent_wins)
        history_win_rates.append(current_win_rate)

        log_file.write(
            f"Episode {episode} End. Total Reward: {total_reward:.2f}, Win: {is_win}\n"
        )

        if episode % 100 == 0:
            print(
                f"Ep {episode:5d} | Score: {total_reward:6.2f} | Win Rate: {current_win_rate*100:3.0f}%"
            )

    # 학습 종료 후 그래프 저장
    plot_results(history_rewards, history_win_rates)


def test(env, agent, args):
    """테스트 모드 실행"""
    print("Start Test Simulation...")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Test Game Over. Total Reward: {total_reward}")
