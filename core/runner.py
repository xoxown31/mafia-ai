import threading
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from core.agent.llmAgent import LLMAgent
from config import Role, EventType, Phase

if TYPE_CHECKING:
    from core.logger import LogManager
    from core.agent.rlAgent import RLAgent


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

    # 승률 계산을 위한 최근 승리 기록 (에이전트별)
    recent_wins = {pid: [] for pid in rl_agents.keys()}
    recent_mafia_wins = []
    recent_citizen_wins = []
    window_size = 100

    # Helper to convert int ID to string ID
    def id_to_agent(i):
        return f"player_{i}"

    for episode in range(1, args.episodes + 1):
        if stop_event and stop_event.is_set():
            print(f"\n[System] Training stopped by user at Episode {episode}.")
            break

        if hasattr(env, "game") and env.game.logger:
            env.game.logger.set_episode(episode)

        obs_dict, _ = env.reset()

        # RNN 은닉 상태 초기화
        for agent in rl_agents.values():
            if hasattr(agent, "reset_hidden"):
                agent.reset_hidden()

        obs_dict, _ = env.reset()
        done = False

        # 에피소드별 보상 추적
        episode_rewards = {pid: 0.0 for pid in rl_agents.keys()}
        is_wins = {pid: False for pid in rl_agents.keys()}

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
        train_metrics = {"Mafia": {"loss": [], "entropy": []}, "Citizen": {"loss": [], "entropy": []}}
        
        for agent in rl_agents.values():
            if hasattr(agent, "update"):
                res = agent.update()
                if res and isinstance(res, dict):
                    role_key = "Mafia" if agent.role == Role.MAFIA else "Citizen"
                    if "loss" in res:
                        train_metrics[role_key]["loss"].append(res["loss"])
                    if "entropy" in res:
                        train_metrics[role_key]["entropy"].append(res["entropy"])

        # 승률 및 로깅 (대표 에이전트 또는 전체)
        metrics = {}
        
        # 1. Agent Tab (학습 현황)
        for pid in rl_agents.keys():
            recent_wins[pid].append(1 if is_wins[pid] else 0)
            if len(recent_wins[pid]) > window_size:
                recent_wins[pid].pop(0)

            win_rate = (
                sum(recent_wins[pid]) / len(recent_wins[pid])
                if recent_wins[pid]
                else 0.0
            )

            # 개별 에이전트 메트릭 추가
            metrics[f"Agent_{pid}/Reward_Total"] = episode_rewards[pid]
            metrics[f"Agent_{pid}/Win_Rate"] = win_rate

        # 팀/역할별 학습 지표
        for role_key in ["Mafia", "Citizen"]:
            if train_metrics[role_key]["loss"]:
                metrics[f"Train/{role_key}_Loss"] = sum(train_metrics[role_key]["loss"]) / len(train_metrics[role_key]["loss"])
            if train_metrics[role_key]["entropy"]:
                metrics[f"Train/{role_key}_Entropy"] = sum(train_metrics[role_key]["entropy"]) / len(train_metrics[role_key]["entropy"])

        # 2. Game Tab (게임 통계)
        if hasattr(env, "game"):
            game = env.game
            metrics["Game/Duration"] = game.day
            
            # 승률 밸런스
            # 이번 게임의 승리 팀 확인
            winner_team = "None"
            # RL 에이전트 중 하나라도 승리했으면 그 팀이 승리한 것으로 간주 (모든 플레이어가 RL이 아닐 수도 있으므로)
            # 하지만 여기서는 RL 에이전트들의 승패만 기록되어 있음.
            # env.game.check_game_over()를 사용하여 정확한 승리 팀을 파악하는 것이 좋음.
            # 하지만 이미 게임이 끝났으므로, is_wins 정보를 활용.
            
            # 마피아 팀 승리 여부 확인
            mafia_won = False
            citizen_won = False
            
            # 전체 에이전트(all_agents)를 순회하며 확인
            for pid, agent in all_agents.items():
                # RL 에이전트인 경우 is_wins에서 확인
                if pid in is_wins:
                    if is_wins[pid]:
                        if agent.role == Role.MAFIA:
                            mafia_won = True
                        else:
                            citizen_won = True
                # RL 에이전트가 아닌 경우 (LLM 등) - infos나 game state에서 확인해야 함
                # 여기서는 RL 에이전트가 적어도 한 명씩은 각 팀에 있다고 가정하거나,
                # 단순히 RL 에이전트들의 승패로 팀 승패를 결정.
                # 만약 RL 에이전트가 없는 팀이 있다면? -> 보통 학습 시에는 양 팀에 RL 에이전트를 배치함.
            
            # 만약 RL 에이전트만으로 판단이 어렵다면, game history의 마지막 이벤트를 확인
            last_event = game.history[-1] if game.history else None
            if last_event and last_event.phase == Phase.GAME_END:
                # value가 True면 시민 승리, False면 마피아 승리 (game.py check_game_over 참조)
                citizen_won_game = last_event.value
                if citizen_won_game:
                    citizen_won = True
                    mafia_won = False
                else:
                    mafia_won = True
                    citizen_won = False
            
            recent_mafia_wins.append(1 if mafia_won else 0)
            recent_citizen_wins.append(1 if citizen_won else 0)
            
            if len(recent_mafia_wins) > window_size: recent_mafia_wins.pop(0)
            if len(recent_citizen_wins) > window_size: recent_citizen_wins.pop(0)
            
            metrics["Game/Mafia_WinRate"] = sum(recent_mafia_wins) / len(recent_mafia_wins) if recent_mafia_wins else 0.0
            metrics["Game/Citizen_WinRate"] = sum(recent_citizen_wins) / len(recent_citizen_wins) if recent_citizen_wins else 0.0

            # 상세 행동 지표
            mafia_kill_attempts = 0
            doctor_saves = 0
            police_investigations = 0
            police_finds = 0
            lynch_executions = 0
            mafia_lynched = 0
            citizen_lynched = 0 # Wrong lynch
            
            night_events = [e for e in game.history if e.phase == Phase.NIGHT]
            # 날짜별로 그룹화하여 처리
            for d in range(1, game.day + 1):
                day_events = [e for e in night_events if e.day == d]
                kill_event = next((e for e in day_events if e.event_type == EventType.KILL), None)
                protect_event = next((e for e in day_events if e.event_type == EventType.PROTECT), None)
                
                if kill_event:
                    mafia_kill_attempts += 1
                    if protect_event and kill_event.target_id == protect_event.target_id:
                        doctor_saves += 1
                
                police_events = [e for e in day_events if e.event_type == EventType.POLICE_RESULT]
                for pe in police_events:
                    police_investigations += 1
                    if pe.value == Role.MAFIA:
                        police_finds += 1
            
            execute_events = [e for e in game.history if e.event_type == EventType.EXECUTE and e.target_id != -1]
            for ee in execute_events:
                lynch_executions += 1
                target = game.players[ee.target_id]
                if target.role == Role.MAFIA:
                    mafia_lynched += 1
                else:
                    citizen_lynched += 1
            
            if mafia_kill_attempts > 0:
                metrics["Action/Doctor_Save_Rate"] = doctor_saves / mafia_kill_attempts
            else:
                metrics["Action/Doctor_Save_Rate"] = 0.0
                
            if police_investigations > 0:
                metrics["Action/Police_Find_Rate"] = police_finds / police_investigations
            else:
                metrics["Action/Police_Find_Rate"] = 0.0
                
            if lynch_executions > 0:
                metrics["Vote/Mafia_Lynch_Rate"] = mafia_lynched / lynch_executions
                metrics["Vote/Wrong_Lynch_Rate"] = citizen_lynched / lynch_executions
            else:
                metrics["Vote/Mafia_Lynch_Rate"] = 0.0
                metrics["Vote/Wrong_Lynch_Rate"] = 0.0

        # 대표 에이전트 (첫 번째) - 호환성 유지
        rep_pid = list(rl_agents.keys())[0]
        rep_win_rate = (
            sum(recent_wins[rep_pid]) / len(recent_wins[rep_pid])
            if recent_wins[rep_pid]
            else 0.0
        )

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
                wr = (
                    sum(recent_wins[pid]) / len(recent_wins[pid])
                    if recent_wins[pid]
                    else 0.0
                )
                log_str += f" | Ag {pid} R:{episode_rewards[pid]:6.2f} W:{wr*100:3.0f}%"
            print(log_str)

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

        obs_dict, _ = env.reset()

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
