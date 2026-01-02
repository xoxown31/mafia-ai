"""
LogManager: 통합 로그 및 모니터링 시스템

주요 기능:
1. JSONL 로깅: GameEvent를 .jsonl 형식으로 기록
2. TensorBoard 통합: 학습 메트릭을 실시간 모니터링
3. 내러티브 해석: 이벤트를 자연어 문장으로 변환 (GUI/LLM용)
"""

import os
import json
import yaml
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from state import GameEvent
from config import Role, Phase, EventType


class LogManager:
    """게임 이벤트 로깅 및 해석 매니저"""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        write_mode: bool = True,
    ):
        """
        Args:
            experiment_name: 실험 이름 (예: "ppo_mlp_20231231")
            log_dir: 로그 저장 디렉토리
            use_tensorboard: TensorBoard 사용 여부 (학습 에이전트가 없으면 False)
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard and write_mode

        self.narrative_templates = self._load_narrative_templates()

        if write_mode:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.log_dir / f"{experiment_name}_{timestamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)

            self.jsonl_path = self.session_dir / "events.jsonl"
            self.jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")

            self.writer = None
            if self.use_tensorboard:
                tensorboard_dir = self.session_dir / "tensorboard"
                self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
                print(f"  - TensorBoard: {tensorboard_dir}")

            print(f"[LogManager] Initialized: {self.session_dir}")
        else:
            self.session_dir = None
            self.jsonl_file = None
            self.writer = None

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
            "CANCEL": "Day {day} | 처형이 부결되었습니다.",
            "KILL": "Night {day} | Player {target_id}가 마피아에게 살해당했습니다.",
            "PROTECT": "Night {day} | 의사가 Player {target_id}를 보호했습니다.",
            "POLICE_RESULT": "Night {day} | 경찰이 Player {target_id}를 조사: {role_name}",
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

    def log_event(self, event: GameEvent):
        """GameEvent를 JSONL 형식으로 기록"""
        event_json = event.model_dump_json(exclude_none=False)
        self.jsonl_file.write(event_json + "\n")
        self.jsonl_file.flush()

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
            self.writer.add_scalar(f"Metrics/{key}", value, episode)

    def interpret_event(self, event: GameEvent) -> str:
        """
        GameEvent를 자연어 문장으로 변환 (데이터 주도 방식)

        이 메서드는 GUI 리플레이와 LLM 에이전트의 프롬프트 생성에 사용됩니다.
        로그 파일에는 저장되지 않습니다.

        Args:
            event: 해석할 게임 이벤트

        Returns:
            자연어 문장
        """

        if event.phase == Phase.GAME_END:
            result = "시민 팀 승리!" if event.value else "마피아 팀 승리!"
            return f"게임 종료 {result}"

        # 1. 이벤트 타입과 템플릿 키 매핑
        type_to_key = {
            EventType.SYSTEM_MESSAGE: "SYSTEM_MESSAGE",
            EventType.KILL: "KILL",
            EventType.PROTECT: "PROTECT",
            EventType.POLICE_RESULT: "POLICE_RESULT",
        }

        # 2. 특수 로직이 필요한 CLAIM 처리
        template_key = type_to_key.get(event.event_type)

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
            if event.target_id == -1:
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
            self._get_role_korean_name(event.value)
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
    def _get_role_korean_name(role: Role) -> str:
        """역할의 한국어 이름 반환"""
        role_names = {
            Role.CITIZEN: "시민",
            Role.POLICE: "경찰",
            Role.DOCTOR: "의사",
            Role.MAFIA: "마피아",
        }
        return role_names.get(role, str(role))

    def load_events(self) -> List[GameEvent]:
        """JSONL 파일에서 모든 이벤트 로드"""
        events = []
        if not self.jsonl_path.exists():
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
            self.jsonl_file.close()
        if self.writer:
            self.writer.close()
        print(f"[LogManager] Closed: {self.session_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
