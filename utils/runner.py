from utils.visualize import plot_results


def train(env, agent, args, log_file):
    """학습 모드 실행"""
    print(f"Start Training ({args.agent.upper()}) for {args.episodes} episodes...")

    history_rewards = []
    history_win_rates = []
    recent_wins = []

    for episode in range(1, args.episodes + 1):
        log_file.write(f"\n{'='*20} Episode {episode} Start {'='*20}\n")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        is_win = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            # 에이전트별 데이터 저장
            if args.agent == "ppo":
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
            elif args.agent == "reinforce":
                agent.rewards.append(reward)

            # 승리 판정 (보상이 5.0 초과면 승리라고 가정)
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
