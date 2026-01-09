"""
LogManager: 통합 로그 및 모니터링 시스템

주요 기능:
1. JSONL 로깅: GameEvent를 .jsonl 형식으로 기록 (분할 저장 지원)
2. TensorBoard 통합: 학습 메트릭을 실시간 모니터링
3. 내러티브 해석: 이벤트를 자연어 문장으로 변환 (GUI/LLM용)
"""

import os
import json
import yaml
import logging
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from core.engine.state import GameEvent
from config import Role, Phase, EventType, config


class LogManager:
    """게임 이벤트 로깅 및 해석 매니저"""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        write_mode: bool = True,
        overwrite: bool = False,
        split_interval: int = 1000,  # [수정] 파일 분할 주기 추가 (기본 1000판)
    ):
        """
        Args:
            experiment_name: 실험 이름 (예: "ppo_mlp_20231231")
            log_dir: 로그 저장 디렉토리
            use_tensorboard: TensorBoard 사용 여부 (학습 에이전트가 없으면 False)
            split_interval: 몇 에피소드마다 로그 파일을 나눌지 설정
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard and write_mode
        self.write_mode = write_mode  # [수정] 멤버 변수로 저장
        self.current_episode = 1
        self.split_interval = split_interval  # [수정] 설정 저장

        self.narrative_templates = self._load_narrative_templates()
        self.jsonl_file = None
        self.jsonl_path = None

        if self.write_mode:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.log_dir / f"{experiment_name}_{timestamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)

            self._open_log_file(1)

            self.writer = None
            if self.use_tensorboard:
                tensorboard_dir = self.session_dir / "tensorboard"
                self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
                self._setup_tensorboard_layout()
                print(f"  - TensorBoard: {tensorboard_dir}")

            print(f"[LogManager] Initialized: {self.session_dir}")
        else:
            self.session_dir = None
            self.writer = None
        
        # [Logger Setup]
        self.logger = logging.getLogger("MafiaLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent propagation to root logger

        if not self.logger.handlers:
            # File Handler
            if self.session_dir:
                log_file = self.session_dir / "system.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                self.logger.addHandler(file_handler)

            # Console Handler
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)

    def _open_log_file(self, start_episode: int):
        """[추가] 새로운 로그 파일을 엽니다."""
        if not self.write_mode:
            return
        if self.jsonl_file and not self.jsonl_file.closed:
            self.jsonl_file.flush()
            self.jsonl_file.close()
        # 파일명 형식: events_ep{시작에피소드}.jsonl
        # 예: events_ep1.jsonl, events_ep1001.jsonl
        filename = f"events_ep{start_episode}.jsonl"
        self.jsonl_path = self.session_dir / filename
        self.jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")
        print(f"[LogManager] Created new log file: {filename}")

    def _setup_tensorboard_layout(self):
        """TensorBoard Custom Scalars 레이아웃 설정"""
        if not self.writer:
            return

        # Individual Agent Charts (Separate charts per agent)
        individual_charts = {}
        for i in range(config.game.PLAYER_COUNT):
            individual_charts[f"Agent {i} Reward"] = [
                "Multiline",
                [f"Agent_{i}/Reward_Total"],
            ]

        layout = {
            "Summary Dashboard": {
                "Win Rates": [
                    "Multiline",
                    ["Game/Mafia_WinRate", "Game/Citizen_WinRate"],
                ],
                "Team Avg Reward": [
                    "Multiline",
                    ["Reward/Mafia_Avg", "Reward/Citizen_Avg"],
                ],
                "Total Reward": ["Multiline", ["Reward/Total"]],
                "Game Duration": [
                    "Multiline",
                    [
                        "Game/Duration",
                        "Game/Avg_Day_When_Mafia_Wins",
                        "Game/Avg_Day_When_Citizen_Wins",
                    ],
                ],
            },
            "Training Details": {
                "Team Loss": ["Multiline", ["Train/Mafia_Loss", "Train/Citizen_Loss"]],
                "Team Entropy": [
                    "Multiline",
                    ["Train/Mafia_Entropy", "Train/Citizen_Entropy"],
                ],
                "Policy Trust (KL)": [
                    "Multiline",
                    ["Train/Mafia_ApproxKL", "Train/Citizen_ApproxKL"],
                ],
                "Clip Fraction": [
                    "Multiline",
                    ["Train/Mafia_ClipFrac", "Train/Citizen_ClipFrac"],
                ],
            },
            "Individual Agents": individual_charts,
            "Role Performance": {
                "Doctor Stats": [
                    "Multiline",
                    ["Action/Doctor_Save_Rate", "Action/Doctor_Self_Heal_Rate"],
                ],
                "Police Stats": ["Multiline", ["Action/Police_Find_Rate"]],
                "Mafia Stats": ["Multiline", ["Action/Mafia_Kill_Success_Rate"]],
                "Citizen Survival": ["Multiline", ["Game/Citizen_Survival_Rate"]],
            },
            "Behavior Analysis": {
                "Voting Behavior": [
                    "Multiline",
                    ["Vote/Abstain_Rate", "Game/Execution_Frequency"],
                ],
                "Voting Accuracy": [
                    "Multiline",
                    ["Vote/Citizen_Accuracy_Rate", "Vote/Mafia_Betrayal_Rate"],
                ],
                "Execution Outcomes": [
                    "Multiline",
                    ["Vote/Mafia_Lynch_Rate", "Vote/Citizen_Sacrifice_Rate"],
                ],
            },
        }

        self.writer.add_custom_scalars(layout)

    def log_histograms(self, episode: int, agent_id: int, tag: str, values: Any):
        """
        히스토그램 로깅
        Args:
            episode: 현재 에피소드
            agent_id: 에이전트 ID
            tag: 데이터 태그 (예: 'Action/Probs')
            values: 데이터 값 (Tensor or Numpy array)
        """
        if self.writer:
            self.writer.add_histogram(
                f"Agent_{agent_id}/{tag}", values, global_step=episode
            )

    def _load_narrative_templates(self) -> Dict[str, str]:
        """YAML에서 내러티브 템플릿 로드"""
        template_path = Path(__file__).parent / "narrative_templates.yaml"

        # 기본 템플릿
        default_templates = {
            "SYSTEM_MESSAGE": "Day {day} | Player {target_id}의 직업은 {role_name}입니다.",
            "CLAIM_SELF": "Day {day} | Player {actor_id}는 자신이 {role_name}라고 주장했습니다.",
            "CLAIM_OTHER": "Day {day} | Player {actor_id}는 Player {target_id}가 {role_name}라고 주장했습니다.",
            "VOTE": "Day {day} | Player {actor_id}가 Player {target_id}에게 투표했습니다.",
            "ABSTAIN": "Day {day} | Player {actor_id}는 기권하였습니다.",
            "EXECUTE": "Day {day} | Player {target_id}가 처형되었습니다. (역할: {role_name})",
            "CANCEL": "Day {day} | 투표가 무산되어 아무도 처형되지 않았습니다.",
            "KILL": "Night {day} | Player {target_id}가 마피아에게 살해당했습니다.",
            "PROTECT": "Night {day} | 의사가 Player {target_id}를 보호했습니다.",
            "POLICE_RESULT": "Night {day} | 경찰이 Player {target_id}를 조사: {role_name}",
            "NIGHT_RESULT": "Day {day} | 지난 밤 Player {target_id}가 사망했습니다.",
            "NIGHT_RESULT_NONE": "Day {day} | 지난 밤 아무도 사망하지 않았습니다.",
            "SILENCE": "Day {day} | Player {actor_id}가 침묵했습니다.",
        }

        # YAML 파일이 있으면 로드
        if template_path.exists():
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    yaml_templates = yaml.safe_load(f)
                    if yaml_templates:
                        default_templates.update(yaml_templates)
            except Exception as e:
                print(f"[LogManager] Warning: Failed to load narrative templates: {e}")

        return default_templates

    def set_episode(self, episode: int):
        self.current_episode = episode

        # [수정] 파일 로테이션 체크
        # 예: interval=1000일 때, 1001, 2001, 3001... 에피소드에서 새 파일 생성
        if self.write_mode and self.split_interval > 0 and episode > 1:
            if (episode - 1) % self.split_interval == 0:
                self._open_log_file(episode)

    def log_event(self, event: GameEvent, custom_episode: int = None):
        """GameEvent를 JSONL 형식으로 기록"""
        if self.jsonl_file:
            # 1. 이벤트를 딕셔너리로 변환
            data = event.dict() if hasattr(event, "dict") else event.model_dump()

            # 2. episode 필드 주입 (들어온 값이 있으면 그걸 쓰고, 없으면 기본값 사용)
            if custom_episode is not None:
                data["episode"] = custom_episode
            else:
                data["episode"] = self.current_episode

            # 3. 파일 쓰기
            self.jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            # [최적화] 매 이벤트마다 flush하면 I/O 병목으로 다운될 수 있어 제거함.
            # 대신 log_metrics 호출 시(에피소드 종료 시) flush 합니다.
            # self.jsonl_file.flush()

    def log_metrics(
        self,
        episode: int,
        total_reward: float,
        is_win: bool,
        win_rate: Optional[float] = None,
        **kwargs,
    ):
        """TensorBoard에 학습 메트릭 기록 (TensorBoard 사용 시에만)"""
        if not self.use_tensorboard or self.writer is None:
            return

        self.writer.add_scalar("Reward/Total", total_reward, episode)
        self.writer.add_scalar("Win/IsWin", 1 if is_win else 0, episode)

        if win_rate is not None:
            self.writer.add_scalar("Win/Rate", win_rate, episode)

        # 추가 메트릭
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, episode)

        # [추가] 에피소드가 끝날 때마다 파일 버퍼를 비워 안전하게 저장
        if self.jsonl_file:
            self.jsonl_file.flush()

    def interpret_event(self, event: GameEvent) -> str:
        """
        GameEvent를 자연어 문장으로 변환 (데이터 주도 방식)
        """

        if event.phase == Phase.GAME_END:
            result = "시민 팀 승리!" if event.value else "마피아 팀 승리!"
            return f"게임 종료 {result}"

        # 1. 이벤트 타입과 템플릿 키 매핑
        type_to_key = {
            EventType.KILL: "KILL",
            EventType.PROTECT: "PROTECT",
            EventType.POLICE_RESULT: "POLICE_RESULT",
        }

        # 2. 특수 로직이 필요한 CLAIM 처리
        template_key = type_to_key.get(event.event_type)

        if event.event_type == EventType.SYSTEM_MESSAGE:
            if event.value == None:
                if event.target_id == -1:
                    template_key = "NIGHT_RESULT_NONE"
                else:
                    template_key = "NIGHT_RESULT"
            else:
                template_key = "SYSTEM_MESSAGE"

        if event.event_type == EventType.CLAIM:
            if event.value is None:
                template_key = "SILENCE"
            elif event.target_id is None or event.target_id == event.actor_id:
                template_key = "CLAIM_SELF"
            else:
                template_key = "CLAIM_OTHER"
        if event.event_type == EventType.VOTE:
            if event.target_id == -1 or event.target_id is None:
                template_key = "ABSTAIN"
            else:
                template_key = "VOTE"
        if event.event_type == EventType.EXECUTE:
            if event.target_id == -1 or event.value is None:
                template_key = "CANCEL"
            else:
                template_key = "EXECUTE"

        if not template_key:
            return f"[Unknown Event] {event.event_type}"

        # 3. 템플릿 가져오기 및 포맷팅
        template = self.narrative_templates.get(template_key, "")
        if not template:
            return f"[No Template] {template_key}"

        role_name = (
            self.get_role_korean_name(event.value)
            if isinstance(event.value, Role)
            else ""
        )

        return template.format(
            day=event.day,
            actor_id=event.actor_id,
            target_id=event.target_id if event.target_id is not None else -1,
            role_name=role_name,
        )

    @staticmethod
    def get_role_korean_name(role: Role) -> str:
        """역할의 한국어 이름 반환"""
        role_names = {
            Role.CITIZEN: "시민",
            Role.POLICE: "경찰",
            Role.DOCTOR: "의사",
            Role.MAFIA: "마피아",
        }
        return role_names.get(role, str(role))

    def load_events(self) -> List[GameEvent]:
        """JSONL 파일에서 모든 이벤트 로드 (현재 활성화된 파일만 로드됨)"""
        events = []
        if self.jsonl_path is None or not self.jsonl_path.exists():
            return events

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event_dict = json.loads(line)
                    events.append(GameEvent(**event_dict))
        return events

    def load_events_by_episode(self, episode_num: int) -> List[GameEvent]:
        """특정 에피소드의 이벤트만 로드 (향후 구현 가능)"""
        # TODO: 에피소드 구분자가 필요한 경우 구현
        raise NotImplementedError("Episode filtering not yet implemented")

    def close(self):
        """리소스 정리"""
        if self.jsonl_file and not self.jsonl_file.closed:
            self.jsonl_file.flush()
            self.jsonl_file.close()
        if self.writer:
            self.writer.close()
        print(f"[LogManager] Closed: {self.session_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
