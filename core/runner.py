from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.logger import LogManager


def train(env, agent, args, logger: 'LogManager'):
    """
    학습 모드 실행 - 학습 가능 에이전트가 있을 때만 update() 호출
    
    LogManager를 통해 JSONL 및 TensorBoard에 로깅합니다.
    """
    algorithm_name = getattr(agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(agent, 'backbone', 'mlp').upper()
    
    # 학습 가능 여부 확인
    is_trainable = hasattr(agent, 'update') and callable(getattr(agent, 'update'))
    
    if is_trainable:
        print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")
    else:
        print(f"Start Simulation ({algorithm_name}) for {args.episodes} episodes... (학습 불가)")

    # 승률 계산을 위한 최근 승리 기록
    recent_wins = []
    window_size = 100

    for episode in range(1, args.episodes + 1):
        # RNN 은닉 상태 초기화 (있는 경우만)
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        is_win = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)

            # 보상 저장 (store_reward 메서드가 있는 경우만)
            if hasattr(agent, 'store_reward'):
                agent.store_reward(reward, done)

            # 승리 판정
            if done and reward > 5.0:
                is_win = True

            obs = next_obs
            total_reward += reward

        # 에피소드 종료 후 학습 (학습 가능한 경우만)
        if is_trainable:
            agent.update()

        # 승률 계산
        recent_wins.append(1 if is_win else 0)
        if len(recent_wins) > window_size:
            recent_wins.pop(0)
        current_win_rate = sum(recent_wins) / len(recent_wins)

        # LogManager를 통한 메트릭 기록
        logger.log_metrics(
            episode=episode,
            total_reward=total_reward,
            is_win=is_win,
            win_rate=current_win_rate
        )

        if episode % 100 == 0:
            mode = "Training" if is_trainable else "Simulation"
            print(
                f"[{mode}] Ep {episode:5d} | Score: {total_reward:6.2f} | Win Rate: {current_win_rate*100:3.0f}%"
            )

    if is_trainable:
        print(f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}")
    else:
        print(f"\n시뮬레이션 완료. 로그 확인: {logger.session_dir}")


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
