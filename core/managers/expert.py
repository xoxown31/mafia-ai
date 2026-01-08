import json
import os
import numpy as np
from typing import Any, Dict
from pathlib import Path

class ExpertDataManager:
    def __init__(self, save_dir: Path):
        """
        save_dir: LogManager가 생성한 session_dir (Path 객체)
        """
        self.save_dir = save_dir
        self.save_path = self.save_dir / "train_set.jsonl"
        
        # 단일 환경이므로 복잡한 버퍼링 없이 바로 리스트에 담았다가 flush해도 됨
        # 하지만 확장성을 위해 episode_id 구조는 유지
        self.episode_buffers: Dict[int, Dict[int, Any]] = {}
        
        # 파일 초기화 (없으면 생성)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def record_turn(self, episode_id: int, player_id: int, observation: np.ndarray, action: Any):
        """한 턴의 (Observation, Action) 쌍을 기록"""
        
        # 1. 버퍼 초기화
        if episode_id not in self.episode_buffers:
            self.episode_buffers[episode_id] = {i: {'obs': [], 'acts': []} for i in range(8)}

        # 2. Observation (Numpy -> List)
        if isinstance(observation, np.ndarray):
            obs_list = observation.tolist()
        else:
            obs_list = list(observation)
        
        # 3. Action (Object or Vector -> Int List)
        target_idx = 8 # Default: No Action
        role_idx = 0
        
        # 객체(GameAction)인 경우
        if hasattr(action, 'target_id'):
            target_idx = action.target_id if action.target_id != -1 else 8
            if action.claim_role:
                role_idx = action.claim_role.value
        # 벡터([target, role])인 경우
        elif isinstance(action, (list, np.ndarray, tuple)):
            target_idx = int(action[0])
            role_idx = int(action[1])
            if target_idx == -1: target_idx = 8

        # 4. 버퍼에 추가
        self.episode_buffers[episode_id][player_id]['obs'].append(obs_list)
        self.episode_buffers[episode_id][player_id]['acts'].append([target_idx, role_idx])

    def flush_episode(self, episode_id: int):
        """에피소드 종료 시 파일에 저장 (Append)"""
        if episode_id not in self.episode_buffers:
            return

        buffer = self.episode_buffers[episode_id]
        
        try:
            with open(self.save_path, 'a', encoding='utf-8') as f:
                for p_id in range(8):
                    if len(buffer[p_id]['obs']) > 0:
                        entry = {
                            "episode_id": episode_id,
                            "player_id": p_id,
                            "obs": buffer[p_id]['obs'],
                            "acts": buffer[p_id]['acts']
                        }
                        f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[DataManager Error] Flush failed: {e}")
        
        # 메모리 해제
        del self.episode_buffers[episode_id]