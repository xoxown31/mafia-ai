import threading
import os
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from core.agents.llm_agent import LLMAgent
from config import Role, EventType, Phase, config
from core.managers.stats import StatsManager

if TYPE_CHECKING:
    from core.managers.logger import LogManager
    from core.agents.rl_agent import RLAgent


def train(
    env,
    rl_agents: Dict[int, Any],
    all_agents: Dict[int, Any],
    args,
    logger: "LogManager",
    stop_event: Optional[threading.Event] = None,
):
    """
    학습 모드 실행 - 다중 에이전트 지원

    Args:
        env: MafiaEnv 인스턴스
        rl_agents: {player_id: RLAgent} 형태의 딕셔너리 (학습 대상)
        all_agents: {player_id: Agent} 형태의 딕셔너리 (모든 에이전트)
        args: 실행 인자
        logger: LogManager
    """
    # 첫 번째 에이전트 기준으로 정보 출력 (대표)
    if not rl_agents:
        print("No RL agents to train.")
        return

    first_agent = next(iter(rl_agents.values()))
    algorithm_name = getattr(first_agent, "algorithm", "RL").upper()
    backbone_name = getattr(first_agent, "backbone", "lstm").upper()

    print(
        f"Start Training ({algorithm_name}+{backbone_name}) for {args.episodes} episodes..."
    )
    print(f"Active RL Agents: {list(rl_agents.keys())}")

    # 통계 매니저 초기화
    stats_manager = StatsManager()

    # Helper to convert int ID to string ID
    def id_to_agent(i):
        return f"player_{i}"

    for episode in range(1, args.episodes + 1):
        if stop_event and stop_event.is_set():
            print(f"\n[System] Training stopped by user at Episode {episode}.")
            break

        if hasattr(env, "game") and env.game.logger:
            env.game.logger.set_episode(episode)

        # RNN 은닉 상태 초기화
        for agent in rl_agents.values():
            if hasattr(agent, "reset_hidden"):
                agent.reset_hidden()

        done = False

        # 에피소드별 보상 추적
        episode_rewards = {pid: 0.0 for pid in rl_agents.keys()}
        is_wins = {pid: False for pid in rl_agents.keys()}
        obs_dict, _ = env.reset()

        while not done:
            if stop_event and stop_event.is_set():
                break

            actions = {}

            # 1. RL Agents Actions
            for pid, agent in rl_agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in obs_dict:
                    action_vector = agent.select_action_vector(obs_dict[agent_key])
                    actions[agent_key] = action_vector

            # 2. Other Agents (LLM/Bot) - Async Execution
            threads = []
            lock = threading.Lock()

            def get_llm_action(player, p_key):
                try:
                    # LLM 에이전트에게 현재 상태 전달
                    # env.game에 직접 접근하는 대신, env.infos 등을 활용하거나
                    # env가 제공하는 인터페이스를 사용해야 하지만, 현재 구조상 game 접근이 불가피함.
                    # 다만, env.agents에 있는 에이전트만 실행하므로 생존 여부는 확인됨.
                    status = env.game.get_game_status(player.id)

                    a = player.get_action(status)
                    with lock:
                        actions[p_key] = a
                except Exception as e:
                    print(f"Error in LLM action: {e}")

            # all_agents 딕셔너리를 사용하여 에이전트 순회
            for pid, agent in all_agents.items():
                p_key = id_to_agent(pid)

                # RL 에이전트는 이미 처리됨
                if pid in rl_agents:
                    continue

                # 살아있는 에이전트만 처리 (env.agents는 살아있는 에이전트 ID 리스트)
                if p_key in env.agents:
                    if isinstance(agent, LLMAgent):
                        t = threading.Thread(
                            target=get_llm_action,
                            args=(agent, p_key),
                            name=f"Thread-{p_key}",
                        )
                        threads.append(t)
                        t.start()
                    else:
                        try:
                            # 일반 봇도 상태 업데이트 필요할 수 있음
                            status = env.game.get_game_status(pid)

                            # 봇 액션 가져오기
                            a = agent.get_action(status)
                            actions[p_key] = a
                        except Exception as e:
                            print(f"Error in Bot action: {e}")

            # 모든 스레드 종료 대기
            for t in threads:
                t.join(timeout=10.0)
                if t.is_alive():
                    print(f"Warning: LLM Agent thread {t.name} timed out.")

            # 3. Environment Step
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            # Check if game is over
            done = any(terminations.values()) or any(truncations.values())

            # 4. Store Rewards & Track Stats
            for pid, agent in rl_agents.items():
                agent_key = id_to_agent(pid)
                if agent_key in rewards:
                    reward = rewards[agent_key]
                    episode_rewards[pid] += reward

                    if hasattr(agent, "store_reward"):
                        # 종료 여부 전달
                        is_term = terminations.get(agent_key, False) or truncations.get(
                            agent_key, False
                        )
                        agent.store_reward(reward, is_term)

                # 승리 여부 확인 (infos에서)
                if agent_key in infos:
                    is_wins[pid] = infos[agent_key].get("win", False)

            obs_dict = next_obs_dict

        if stop_event and stop_event.is_set():
            break

        # 에피소드 종료 후 학습 및 메트릭 수집
        train_metrics = {
            "Mafia": {"loss": [], "entropy": []},
            "Citizen": {"loss": [], "entropy": []},
        }

        for agent in rl_agents.values():
            if hasattr(agent, "update"):
                res = agent.update()
                if res and isinstance(res, dict):
                    role_key = "Mafia" if agent.role == Role.MAFIA else "Citizen"
                    if "loss" in res:
                        train_metrics[role_key]["loss"].append(res["loss"])
                    if "entropy" in res:
                        train_metrics[role_key]["entropy"].append(res["entropy"])

        # StatsManager를 통해 통계 계산
        metrics = stats_manager.calculate_stats(
            env=env,
            rl_agents=rl_agents,
            all_agents=all_agents,
            episode_rewards=episode_rewards,
            is_wins=is_wins,
            train_metrics=train_metrics,
        )

        # 대표 에이전트 (첫 번째) - 호환성 유지
        rep_pid = list(rl_agents.keys())[0]
        # StatsManager에서 계산된 값 사용
        rep_win_rate = metrics.get(f"Agent_{rep_pid}/Win_Rate", 0.0)

        logger.log_metrics(
            episode=episode,
            total_reward=episode_rewards[rep_pid],
            is_win=is_wins[rep_pid],
            win_rate=rep_win_rate,
            **metrics,
        )

        if episode % 100 == 0:
            # 모든 에이전트 상태 출력
            log_str = f"[Training] Ep {episode:5d}"
            for pid in rl_agents.keys():
                wr = metrics.get(f"Agent_{pid}/Win_Rate", 0.0)
                log_str += f" | Ag {pid} R:{episode_rewards[pid]:6.2f} W:{wr*100:3.0f}%"
            print(log_str)

    print("\n[System] Saving trained models...")

    model_dir = getattr(config.paths, "MODEL_DIR", "./models")
    os.makedirs(model_dir, exist_ok=True)

    # 모든 RL 에이전트 모델 파일로 저장
    for pid, agent in rl_agents.items():
        # 파일명 예: agent_0_ppo_lstm.pt
        save_path = os.path.join(
            model_dir, f"agent_{pid}_{agent.algorithm}_{agent.backbone}.pt"
        )
        # RLAgent 클래스의 save 메서드 호출
        if hasattr(agent, "save"):
            agent.save(save_path)
            print(f"Saved Agent {pid} to: {save_path}")

    print(
        f"\n학습 완료. TensorBoard로 결과 확인: tensorboard --logdir={logger.session_dir / 'tensorboard'}"
    )


def test(
    env, all_agents: Dict[int, Any], args, stop_event: Optional[threading.Event] = None
):
    """테스트 모드 실행 - 다중 에이전트 지원"""
    print("Start Test Simulation...")
    obs_dict, _ = env.reset()
    done = False

    episode_rewards = {pid: 0.0 for pid in all_agents.keys()}

    # Helper to convert int ID to string ID
    def id_to_agent(i):
        return f"player_{i}"

    while not done:
        if stop_event and stop_event.is_set():
            print("\n[System] Test stopped by user.")
            break

        actions = {}

        if hasattr(env, "game") and env.game.logger:
            env.game.logger.set_episode(1)

        # Iterate all agents
        for pid, agent in all_agents.items():
            if stop_event and stop_event.is_set():
                print("\n[System] Test stopped by user.")
                return

            agent_key = id_to_agent(pid)

            # Skip dead agents
            if agent_key not in env.agents:
                continue

            # RL Agent
            if hasattr(agent, "select_action_vector") and agent_key in obs_dict:
                action_vector = agent.select_action_vector(obs_dict[agent_key])
                actions[agent_key] = action_vector

            # LLM / Bot Agent
            else:
                # Update status
                status = env.game.get_game_status(pid)
                actions[agent_key] = agent.get_action(status)

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        done = any(terminations.values()) or any(truncations.values())

        for pid in all_agents.keys():
            agent_key = id_to_agent(pid)
            if agent_key in rewards:
                episode_rewards[pid] += rewards[agent_key]

        obs_dict = next_obs_dict
        env.render()

    print("Test Game Over.")
    for pid, score in episode_rewards.items():
        print(f"Agent {pid} Total Reward: {score}")
