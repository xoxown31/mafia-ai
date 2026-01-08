import numpy as np
import os
from typing import Dict, Any, Optional
import threading
import torch
from tqdm import tqdm

from config import Role, config
from core.managers.stats import StatsManager
from core.managers.logger import LogManager
from core.managers.expert import ExpertDataManager


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
    print(f"  - Actual Parallel Games: {actual_num_games}")

    stats_manager = StatsManager()
    total_episodes = args.episodes

    slot_episode_ids = [i + 1 for i in range(actual_num_games)]
    next_global_episode_id = actual_num_games + 1

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

    pbar = tqdm(total=total_episodes, desc="Training", unit="ep")


    while True:
        is_target_still_running = any(eid <= total_episodes for eid in slot_episode_ids)
        
        if completed_episodes >= total_episodes and not is_target_still_running:
            break

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

        if isinstance(infos, dict):
            iterator = [(0, item) for item in infos.values()]
        elif isinstance(infos, list):
            iterator = enumerate(infos)
        else:
            iterator = []

        for flat_idx, info_item in iterator:
            if isinstance(info_item, dict) and "log_events" in info_item:
                
                game_slot_idx = flat_idx // PLAYERS_PER_GAME
                custom_id = slot_episode_ids[game_slot_idx]
                
                if custom_id > total_episodes:
                    continue
                
                for ev_dict in info_item["log_events"]:
                    try:
                        from core.engine.state import GameEvent
                        event_obj = GameEvent(**ev_dict)
                        logger.log_event(event_obj, custom_episode=custom_id)
                    except Exception as e:
                        print(f"[Log Error] {e}")

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
            finished_slot_indices = np.where(dones)[0]
            for slot_idx in finished_slot_indices:
                slot_episode_ids[slot_idx] = next_global_episode_id
                next_global_episode_id += 1
            
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

            pbar.update(num_finished_now)

            # 업데이트 주기 체크
            if (
                completed_episodes - num_finished_now
            ) // 100 != completed_episodes // 100:
                pbar.write("[System] Updating Agents...")
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
    env, 
    all_agents: Dict[int, Any], 
    args, 
    logger: "LogManager" = None, # [수정] Logger 받음
    stop_event: Optional[threading.Event] = None
):
    """
    [데이터 수집 겸용 테스트]
    - 단일 환경(Single Env)에서 순차적으로 실행됩니다.
    - logger가 있으면 자동으로 학습 데이터(train_set.jsonl)를 수집합니다.
    """
    num_episodes = args.episodes
    print(f"=== Start Data Collection / Test (Target: {num_episodes} eps) ===")

    # 1. DataManager 연결
    data_manager = None
    if logger and logger.session_dir:
        data_manager = ExpertDataManager(save_dir=logger.session_dir)
        print(f"  - [Data Collection ON] Saved to: {data_manager.save_path}")
    else:
        print("  - [Data Collection OFF] No logger provided.")

    completed_episodes = 0
    obs, info = env.reset()
    
    # 통계용
    win_counts = {Role.CITIZEN: 0, Role.MAFIA: 0}

    pbar = tqdm(total=num_episodes, desc="Collecting", unit="ep")

    while completed_episodes < num_episodes:
        if stop_event and stop_event.is_set(): break

        # --- [1. 행동 결정] ---
        actions = {}
        
        # 살아있는 에이전트 순회
        # (env.game.players 접근이 가능하다면 활용, 아니면 0~7 루프)
        for p_id in range(8):
            if not env.game.players[p_id].alive:
                continue

            agent = all_agents[p_id]
            p_obs = obs[p_id]
            
            # 행동 결정 (호환성 처리)
            action_obj = None
            action_vector = [8, 0] # Default No-Op

            try:
                # 1. LLM / Rule / Heuristic (GameStatus 필요)
                if hasattr(agent, "get_action"):
                    # 단일 환경이므로 get_game_status 호출 안전
                    game_status = env.get_game_status(p_id)
                    action_obj = agent.get_action(game_status)
                    
                    # 기록용 벡터 변환
                    tgt = action_obj.target_id if action_obj.target_id != -1 else 8
                    role = action_obj.claim_role.value if action_obj.claim_role else 0
                    action_vector = [tgt, role]
                    
                    # 환경에 전달할 객체 저장
                    actions[p_id] = action_obj

                # 2. RL Agent (Vector Obs 필요)
                elif hasattr(agent, "select_action_vector"):
                    action_vector = agent.select_action_vector(p_obs)
                    # RL은 action_vector를 그대로 뱉음.
                    # 하지만 단일 env.step은 보통 dict{int: ActionObject}를 원하거나
                    # env 내부에서 벡터를 해석해줘야 함.
                    # 여기서는 'mafia_env.py'가 벡터를 받아서 처리한다고 가정하거나,
                    # RL Agent가 Action 객체로 변환해주는 래퍼가 필요할 수 있음.
                    # (일단 기존 구조상 RL도 vector를 actions에 넣으면 env가 처리한다고 가정)
                    actions[p_id] = action_vector

                # --- [핵심] 데이터 저장 ---
                if data_manager:
                    # 현재 에피소드 ID (1부터 시작)
                    current_ep_id = completed_episodes + 1
                    data_manager.record_turn(current_ep_id, p_id, p_obs, action_vector)

            except Exception as e:
                print(f"[Error] Agent {p_id}: {e}")

        # --- [2. 환경 진행] ---
        next_obs, rewards, done, info = env.step(actions)

        # --- [3. 로그 저장] ---
        # LogManager는 보통 info['log_events']나 env.game.events를 참조
        if logger:
            # env.game.events에 쌓인 것을 가져와서 로깅 (가장 확실한 방법)
            # LogManager 구현에 따라 다르지만, 보통 step 직후 발생한 이벤트를 처리
            pass 
            # (만약 env.step()에서 info에 log를 담아준다면 아래 코드 사용)
            if isinstance(info, dict) and "log_events" in info:
                 for ev_dict in info["log_events"]:
                     try:
                        from core.engine.state import GameEvent
                        logger.log_event(GameEvent(**ev_dict))
                     except: pass

        obs = next_obs

        # --- [4. 종료 체크] ---
        if done:
            # 결과 기록
            if env.game.winner:
                win_counts[env.game.winner] += 1
            
            # 데이터 파일 쓰기 (Flush)
            if data_manager:
                data_manager.flush_episode(completed_episodes + 1)
            
            # 로그 매니저에게 에피소드 종료 알림 (통계 등)
            if logger:
                # 간단히 승리 여부만 기록
                is_win = (env.game.winner == Role.MAFIA) # 예시
                logger.log_metrics(completed_episodes + 1, total_reward=0, is_win=is_win)
                logger.set_episode(completed_episodes + 2) # 다음 에피소드 번호 세팅

            completed_episodes += 1
            pbar.update(1)
            obs, _ = env.reset()

    print(f"\n=== Test/Collection Finished ===")
    print(f"Results: {win_counts}")
    if data_manager:
        print(f"Expert Data Saved: {data_manager.save_path}")