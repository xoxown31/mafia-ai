import numpy as np
import os
from typing import Dict, Any, Optional
import threading
import torch

from config import Role, config
from core.managers.stats import StatsManager
from core.managers.logger import LogManager


def train(
    env,
    rl_agents: Dict[int, Any],
    all_agents: Dict[int, Any],
    args,
    logger: "LogManager",
    stop_event: Optional[threading.Event] = None,
):
    """
    [수정됨] SuperSuit VectorEnv 전용 병렬 학습 루프
    - 배치 크기 불일치(Shape Mismatch) 원천 차단
    """

    # 1. 플레이어 수 동적 감지 (하드코딩 방지)
    # env.possible_agents가 있으면 사용, 없으면 config나 rl_agents+all_agents 길이로 추론
    if hasattr(env, "possible_agents"):
        PLAYERS_PER_GAME = len(env.possible_agents)
    else:
        PLAYERS_PER_GAME = 8

    # 2. [핵심 수정] 실제 병렬 게임 수 계산
    # 나눗셈(//) 대신 실제 range로 생성되는 인덱스 개수를 세어야 함 (나머지 처리)
    # 0번 플레이어 기준으로 몇 개의 게임이 생성되는지 확인
    sample_indices = list(range(0, env.num_envs, PLAYERS_PER_GAME))
    actual_num_games = len(sample_indices)

    print(f"=== Start SuperSuit Parallel Training ===")
    print(f"  - Total Slots (num_envs): {env.num_envs}")
    print(f"  - Players per Game: {PLAYERS_PER_GAME}")
    print(f"  - Actual Parallel Games: {actual_num_games}")
    print(f"  - Target Episodes: {args.episodes}")

    stats_manager = StatsManager()
    total_episodes = args.episodes

    # --- 초기화 ---
    obs, info = env.reset()

    # [수정] RNN Hidden States 초기화 (계산된 actual_num_games 사용)
    for agent in rl_agents.values():
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden(batch_size=actual_num_games)

    completed_episodes = 0

    current_rewards = {}  # 빈 딕셔너리 생성

    # 에이전트마다 자기 할당량을 직접 계산 (len)
    for pid in rl_agents.keys():
        # 내(pid)가 처리해야 할 데이터의 인덱스 리스트 생성
        # 예: 전체 65개 슬롯, 8명 게임 -> 0번 플레이어는 9개, 1번 플레이어는 8개...
        my_indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

        # 실제 내가 받은 데이터 개수 (9 or 8)
        my_batch_size = len(my_indices)

        # 1. RNN Hidden State 초기화 (내 실제 크기에 맞춤)
        if hasattr(rl_agents[pid], "reset_hidden"):
            rl_agents[pid].reset_hidden(batch_size=my_batch_size)

        # 2. 보상 버퍼 초기화 (내 실제 크기에 맞춤)
        # 이렇게 하면 나중에 (9,) 그릇에 (9,) 데이터를 담게 되어 에러가 안 남
        current_rewards[pid] = np.zeros(my_batch_size)

    while completed_episodes < total_episodes:
        if stop_event and stop_event.is_set():
            break

        # --- [1. 행동 결정 (Batch Action)] ---
        all_actions = np.zeros((env.num_envs, 2), dtype=int)

        for pid in range(PLAYERS_PER_GAME):
            # PID별 인덱스 추출
            indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

            # 혹시 모를 인덱스 길이 불일치 방지 (매우 드문 케이스)
            if len(indices) != actual_num_games:
                # 마지막 자투리 배치가 다를 경우를 대비해 0으로 패딩하거나 잘라야 함
                # 하지만 보통 VectorEnv는 맞춰져 있으므로 여기서는 pass
                pass

            if pid in rl_agents:
                agent = rl_agents[pid]

                # 관측 데이터 슬라이싱
                if isinstance(obs, dict):
                    agent_obs = {
                        key: val[indices]
                        for key, val in obs.items()
                        if isinstance(val, (np.ndarray, list))
                    }
                else:
                    agent_obs = obs[indices]

                # 행동 선택
                actions = agent.select_action_vector(agent_obs)

                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)

                # 차원 안전장치
                if len(actions.shape) == 1:
                    pass

                # 배열 할당 (여기도 indices 길이와 actions 길이가 같으므로 안전)
                all_actions[indices] = actions

            elif pid in all_agents:
                # LLM/Rule 기반 스킵
                pass

        # --- [2. 환경 진행 (Step)] ---
        next_obs, rewards, terminations, truncations, infos = env.step(all_actions)

        # --- [3. 보상 저장 및 버퍼 관리] ---
        for pid, agent in rl_agents.items():
            indices = list(range(pid, env.num_envs, PLAYERS_PER_GAME))

            # 배치 데이터 추출
            p_rewards = rewards[indices]
            p_terms = terminations[indices]
            p_truncs = truncations[indices]

            # [문제 해결 구간]
            # current_rewards[pid] (길이 N) += p_rewards (길이 N) -> 차원 일치 보장됨
            current_rewards[pid] += p_rewards

            # 학습용 버퍼 저장
            for i, idx in enumerate(indices):
                is_done = p_terms[i] or p_truncs[i]
                agent.store_reward(p_rewards[i], is_done)

        obs = next_obs

        # --- [4. 종료 체크 및 모델 업데이트] ---
        # 0번 플레이어 기준 종료 체크
        p0_indices = list(range(0, env.num_envs, PLAYERS_PER_GAME))
        dones = np.logical_or(terminations[p0_indices], truncations[p0_indices])
        num_finished_now = np.sum(dones)

        if num_finished_now > 0:
            completed_episodes += num_finished_now
            finished_indices = np.where(dones)[0]

            # 대표 에이전트 평균 보상 로깅
            rep_pid = list(rl_agents.keys())[0]
            avg_reward = np.mean(current_rewards[rep_pid][finished_indices])

            logger.set_episode(int(completed_episodes))
            logger.log_metrics(
                episode=int(completed_episodes), total_reward=avg_reward, is_win=False
            )

            # 끝난 게임의 누적 보상 리셋
            for pid in rl_agents:
                current_rewards[pid][finished_indices] = 0.0

            print(f"Progress: {completed_episodes}/{total_episodes} episodes.")

            # 업데이트 주기 체크
            if (
                completed_episodes - num_finished_now
            ) // 100 != completed_episodes // 100:
                print("[System] Updating Agents...")
                for agent in rl_agents.values():
                    if hasattr(agent, "update"):
                        agent.update()

    # --- 학습 종료 및 저장 ---
    print("\n[System] Saving trained models...")
    model_dir = getattr(config.paths, "MODEL_DIR", "./models")
    os.makedirs(model_dir, exist_ok=True)
    for pid, agent in rl_agents.items():
        save_path = os.path.join(model_dir, f"agent_{pid}_supersuit.pt")
        if hasattr(agent, "save"):
            agent.save(save_path)
            print(f"Saved: {save_path}")


def test(
    env, all_agents: Dict[int, Any], args, stop_event: Optional[threading.Event] = None
):
    """테스트 모드 (단일 환경 호환 유지)"""
    print("Start Test Simulation...")
    obs_dict, _ = env.reset()
    done = False
    episode_rewards = {pid: 0.0 for pid in all_agents.keys()}

    def id_to_agent(i):
        return f"player_{i}"

    while not done:
        if stop_event and stop_event.is_set():
            break
        actions = {}

        for pid, agent in all_agents.items():
            agent_key = id_to_agent(pid)
            if agent_key not in env.agents:
                continue

            # 테스트 모드에서는 단일 관측값이므로 바로 전달
            if hasattr(agent, "select_action_vector") and agent_key in obs_dict:
                action_vector = agent.select_action_vector(obs_dict[agent_key])
                actions[agent_key] = action_vector
            else:
                status = env.game.get_game_status(pid)
                actions[agent_key] = agent.get_action(status)

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())

        for pid in all_agents.keys():
            if id_to_agent(pid) in rewards:
                episode_rewards[pid] += rewards[id_to_agent(pid)]

        obs_dict = next_obs_dict

    print("Test Game Over.")
    for pid, score in episode_rewards.items():
        print(f"Agent {pid} Total Reward: {score}")
