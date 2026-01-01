import threading
from typing import TYPE_CHECKING, Dict, Any, List
from core.agent.llmAgent import LLMAgent

if TYPE_CHECKING:
    from core.logger import LogManager
    from core.agent.rlAgent import RLAgent


def train(env, agents: Dict[int, Any], args, logger: 'LogManager'):
    """
    학습 모드 실행 - 다중 에이전트 지원
    
    Args:
        env: MafiaEnv 인스턴스
        agents: {player_id: RLAgent} 형태의 딕셔너리
        args: 실행 인자
        logger: LogManager
    """
    # 첫 번째 에이전트 기준으로 정보 출력 (대표)
    if not agents:
        print("No RL agents to train.")
        return

    first_agent = next(iter(agents.values()))
    algorithm_name = getattr(first_agent, 'algorithm', args.agent).upper()
    backbone_name = getattr(first_agent, 'backbone', 'mlp').upper()
    
    print(f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes...")
    print(f"Active RL Agents: {list(agents.keys())}")

    # 승률 계산을 위한 최근 승리 기록 (에이전트별)
    recent_wins = {pid: [] for pid in agents.keys()}
    window_size = 100

    # Helper to convert int ID to string ID
    def id_to_agent(i): return f"player_{i}"

    for episode in range(1, args.episodes + 1):
        # RNN 은닉 상태 초기화
        for agent in agents.values():
            if hasattr(agent, 'reset_hidden'):
                agent.reset_hidden()
        
        obs_dict, _ = env.reset()
        done = False
        
        # 에피소드별 보상 추적
        episode_rewards = {pid: 0.0 for pid in agents.keys()}
        is_wins = {pid: False for pid in agents.keys()}

        while not done:
            actions = {}
            
            # 1. RL Agents Actions
            for pid, agent in agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in obs_dict:
                    action_vector = agent.select_action_vector(obs_dict[agent_key])
                    actions[agent_key] = action_vector
            
            # 2. Other Agents (LLM/Bot) - Async Execution
            threads = []
            lock = threading.Lock()
            
            def get_llm_action(player, p_key):
                try:
                    a = player.get_action()
                    with lock:
                        actions[p_key] = a
                except Exception as e:
                    print(f"Error in LLM action: {e}")

            for p in env.game.players:
                p_key = id_to_agent(p.id)
                # RL 에이전트가 아닌 경우만 처리
                if p.id not in agents and p_key not in actions and p.alive:
                    if isinstance(p, LLMAgent):
                        t = threading.Thread(target=get_llm_action, args=(p, p_key))
                        threads.append(t)
                        t.start()
                    else:
                        try:
                            actions[p_key] = p.get_action()
                        except Exception as e:
                            print(f"Error in Bot action: {e}")
            
            for t in threads:
                t.join()

            # 3. Environment Step
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if game is over
            done = any(terminations.values()) or any(truncations.values())

            # 4. Store Rewards & Track Stats
            for pid, agent in agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in rewards:
                    reward = rewards[agent_key]
                    episode_rewards[pid] += reward
                    
                    if hasattr(agent, 'store_reward'):
                        # 종료 여부 전달
                        is_term = terminations.get(agent_key, False) or truncations.get(agent_key, False)
                        agent.store_reward(reward, is_term)
                
                # 승리 여부 확인 (infos에서)
                if agent_key in infos:
                    is_wins[pid] = infos[agent_key].get("win", False)

            obs_dict = next_obs_dict

        # 에피소드 종료 후 학습
        for agent in agents.values():
            if hasattr(agent, 'update'):
                agent.update()

        # 승률 및 로깅 (대표 에이전트 또는 전체)
        for pid in agents.keys():
            recent_wins[pid].append(1 if is_wins[pid] else 0)
            if len(recent_wins[pid]) > window_size:
                recent_wins[pid].pop(0)
        
        # 대표 에이전트 (첫 번째)
        rep_pid = list(agents.keys())[0]
        rep_win_rate = sum(recent_wins[rep_pid]) / len(recent_wins[rep_pid]) if recent_wins[rep_pid] else 0.0
        
        logger.log_metrics(
            episode=episode,
            total_reward=episode_rewards[rep_pid],
            is_win=is_wins[rep_pid],
            win_rate=rep_win_rate
        )

        if episode % 100 == 0:
            print(
                f"[Training] Ep {episode:5d} | Agent {rep_pid} Score: {episode_rewards[rep_pid]:6.2f} | Win Rate: {rep_win_rate*100:3.0f}%"
            )

    print(f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}")


def test(env, agents: Dict[int, Any], args):
    """테스트 모드 실행 - 다중 에이전트 지원"""
    print("Start Test Simulation...")
    obs_dict, _ = env.reset()
    done = False
    
    episode_rewards = {pid: 0.0 for pid in agents.keys()}

    # Helper to convert int ID to string ID
    def id_to_agent(i): return f"player_{i}"

    while not done:
        actions = {}
        
        # RL Agents
        for pid, agent in agents.items():
            agent_key = id_to_agent(pid)
            if agent_key in obs_dict:
                action_vector = agent.select_action_vector(obs_dict[agent_key])
                actions[agent_key] = action_vector
            
        # Other Agents
        for p in env.game.players:
            p_key = id_to_agent(p.id)
            if p.id not in agents and p_key not in actions and p.alive:
                actions[p_key] = p.get_action()

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        
        done = any(terminations.values()) or any(truncations.values())
        
        for pid in agents.keys():
            agent_key = id_to_agent(pid)
            if agent_key in rewards:
                episode_rewards[pid] += rewards[agent_key]
            
        obs_dict = next_obs_dict
        env.render()

    print("Test Game Over.")
    for pid, score in episode_rewards.items():
        print(f"Agent {pid} Total Reward: {score}")
