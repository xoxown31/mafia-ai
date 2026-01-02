import json
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QFileDialog,
    QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer

from state import GameEvent
from config import Role, Phase, EventType
from core.logger import LogManager


class LogViewerTab(QWidget):
    """게임 로그를 자연어로 표시하는 탭 (PyQt6)"""

    def __init__(self, parent):
        super().__init__(parent)
        self.current_log_dir: Optional[Path] = None
        self.events: List[GameEvent] = []
        self.log_manager: Optional[LogManager] = None

        self.base_watch_dir = None
        self.is_monitoring = False
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._monitor_update)

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # === 1. 상단: 디렉토리 선택 영역 ===
        top_frame = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_frame.setLayout(top_layout)

        top_layout.addWidget(QLabel("로그 디렉토리:"))

        self.path_label = QLabel("선택된 디렉토리 없음")
        self.path_label.setFrameStyle(QLabel.Shape.Panel | QLabel.Shadow.Sunken)
        top_layout.addWidget(self.path_label, stretch=1)

        btn_select = QPushButton("디렉토리 선택")
        btn_select.clicked.connect(self._select_directory)
        top_layout.addWidget(btn_select)

        btn_refresh = QPushButton("새로고침")
        btn_refresh.clicked.connect(self._load_logs)
        top_layout.addWidget(btn_refresh)

        layout.addWidget(top_frame)

        # === 2. 필터 프레임 ===
        filter_group = QGroupBox("필터")
        filter_layout = QHBoxLayout()
        filter_group.setLayout(filter_layout)

        # Day 필터
        filter_layout.addWidget(QLabel("Day:"))
        self.day_combo = QComboBox()
        self.day_combo.addItem("전체")
        self.day_combo.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.day_combo)

        # Phase 필터
        filter_layout.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["전체", "낮 토론", "투표", "처형 여부 결정", "밤"])
        self.phase_combo.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.phase_combo)

        # 이벤트 타입 필터
        filter_layout.addWidget(QLabel("이벤트 타입:"))
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(
            ["전체", "주장", "투표", "처형", "살해", "보호", "조사"]
        )
        self.event_type_combo.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.event_type_combo)

        filter_layout.addStretch()
        layout.addWidget(filter_group)

        # === 3. 중앙: 로그 표시 영역 ===
        log_group = QGroupBox("게임 이벤트 로그")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # 폰트 설정
        self.log_text.setStyleSheet(
            "font-family: '맑은 고딕', 'Malgun Gothic'; font-size: 13px;"
        )
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group, stretch=1)

        # === 4. 하단: 통계 정보 ===
        stats_group = QGroupBox("통계")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.stats_label = QLabel("이벤트 로드되지 않음")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

    def _select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "로그 디렉토리 선택", "./logs"
        )

        if directory:
            self.current_log_dir = Path(directory)
            self.path_label.setText(str(self.current_log_dir))
            self._load_logs()

    def select_live(self, base_path_str):
        self.base_watch_dir = Path(base_path_str)

        if not self.base_watch_dir.exists():
            self._show_message(f"경로를 찾을 수 없음: {base_path_str}")
            return

        self.is_monitoring = True
        self.path_label.setText(f"실시간 감시 중... ({base_path_str})")

        # 1초마다 최신 로그를 확인하도록 타이머 시작
        self.monitor_timer.start(1000)

        # 즉시 한 번 실행 (혹시 폴더가 이미 있을 수 있으니)
        self._monitor_update()

    def _monitor_update(self):
        """폴더 내 가장 최신 로그 디렉토리를 찾아 로드"""
        if not self.base_watch_dir:
            return

        try:
            # 1. base_path 안의 하위 폴더들을 모두 찾음
            subdirs = [d for d in self.base_watch_dir.iterdir() if d.is_dir()]
            if not subdirs:
                return

            # 2. 가장 최근에 수정된 폴더 찾기 (방금 실행한 시뮬레이션 폴더)
            latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)

            # 3. 새로운 폴더거나, 내용이 바뀌었으면 로드 진행
            # (단순화를 위해 매번 로드 시도 -> _load_logs 내부에서 파일 읽음)
            self.current_log_dir = latest_dir
            self.path_label.setText(str(self.current_log_dir))

            # 기존 _load_logs 함수를 재활용하여 파일 읽기
            # (silent=True는 오류 메시지 박스를 계속 띄우지 않기 위함)
            self._load_logs(silent=True)

        except Exception as e:
            print(f"Monitoring error: {e}")

    def _load_logs(self, silent=False):
        if not self.current_log_dir:
            self._show_message("디렉토리를 먼저 선택해주세요.")
            return

        jsonl_path = self.current_log_dir / "events.jsonl"
        if not jsonl_path.exists():
            if not silent:
                self._show_message("아직 로그 파일이 생성되지 않았습니다.")
            return

        # LogManager 초기화
        try:
            self.log_manager = LogManager(
                experiment_name="viewer",
                log_dir=str(self.current_log_dir.parent),
                use_tensorboard=False,
                write_mode=False,
            )
        except Exception as e:
            print(f"LogManager 초기화 실패: {e}")
            self.log_manager = None

        # JSONL 파싱
        self.events = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = GameEvent(**data)
                        self.events.append(event)
        except Exception as e:
            self._show_message(f"로그 로드 실패: {e}")
            return

        if self.is_monitoring:
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

        # Day 필터 옵션 업데이트
        days = sorted(set(e.day for e in self.events))
        self.day_combo.blockSignals(True)  # 시그널 잠시 차단
        self.day_combo.clear()
        self.day_combo.addItem("전체")
        self.day_combo.addItems([f"Day {d}" for d in days])
        self.day_combo.blockSignals(False)

        self._apply_filter()

    # 로그 표시
    def _apply_filter(self):
        if not self.events:
            return

        day_filter = self.day_combo.currentText()
        phase_filter = self.phase_combo.currentText()
        event_type_filter = self.event_type_combo.currentText()

        filtered_events = []
        for event in self.events:
            # Day 필터
            if day_filter != "전체":
                day_num = int(day_filter.split()[1])
                if event.day != day_num:
                    continue

            # Phase 필터
            if phase_filter != "전체":
                phase_korean = self._phase_to_korean(event.phase)
                if phase_korean != phase_filter:
                    continue

            # 이벤트 타입 필터
            if event_type_filter != "전체":
                event_type_korean = self._event_type_to_korean(event.event_type)
                if event_type_korean != event_type_filter:
                    continue

            filtered_events.append(event)

        self._display_logs(filtered_events)
        self._update_stats(filtered_events)

    def _display_logs(self, events: List[GameEvent]):
        """필터링된 이벤트를 텍스트 위젯에 표시 (HTML 사용)"""
        self.log_text.clear()

        if not events:
            self.log_text.setPlainText("필터 조건에 맞는 이벤트가 없습니다.")
            return

        grouped = defaultdict(list)
        for event in events:
            key = (event.day, event.phase)
            grouped[key].append(event)

        html_content = ""

        for (day, phase), group_events in sorted(grouped.items()):
            # 헤더
            phase_str = self._phase_to_korean(phase)
            html_content += (
                f"<h3 style='color: #0066cc;'>═══ Day {day} - {phase_str} ═══</h3>"
            )

            # 이벤트들
            for event in group_events:
                event_text = self._format_event(event)
                color = self._get_event_color(event.event_type)

                style = f"color: {color};"
                if event.event_type == EventType.EXECUTE:
                    style += " font-weight: bold;"

                html_content += (
                    f"<div style='margin-left: 10px; {style}'>• {event_text}</div>"
                )

            html_content += "<br>"

        self.log_text.setHtml(html_content)

    def _get_event_color(self, event_type: EventType) -> str:
        color_map = {
            EventType.VOTE: "#ff6600",
            EventType.EXECUTE: "#cc0000",
            EventType.KILL: "#990000",
            EventType.PROTECT: "#009900",
            EventType.POLICE_RESULT: "#6ee2ff",
        }
        return color_map.get(event_type, "#f6f6f8")

    def _format_event(self, event: GameEvent) -> str:
        """이벤트 해석 (LogManager 활용)"""
        if self.log_manager:
            try:
                return self.log_manager.interpret_event(event)
            except:
                pass

        # Fallback (LogManager 실패 시 간단 표시)
        return f"[{event.event_type.name}] Actor: {event.actor_id}, Target: {event.target_id}"

    def _phase_to_korean(self, phase: Phase) -> str:
        phase_map = {
            Phase.DAY_DISCUSSION: "낮 토론",
            Phase.DAY_VOTE: "투표",
            Phase.DAY_EXECUTE: "처형 여부 결정",
            Phase.NIGHT: "밤",
        }
        return phase_map.get(phase, phase.name)

    def _event_type_to_korean(self, event_type: EventType) -> str:
        type_map = {
            EventType.CLAIM: "주장",
            EventType.VOTE: "투표",
            EventType.EXECUTE: "처형",
            EventType.KILL: "살해",
            EventType.PROTECT: "보호",
            EventType.POLICE_RESULT: "조사",
        }
        return type_map.get(event_type, event_type.name)

    def _update_stats(self, events: List[GameEvent]):
        total = len(events)
        if total == 0:
            self.stats_label.setText("이벤트 없음")
            return

        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.event_type] += 1

        stats_parts = [f"총 이벤트: {total}"]
        for event_type, count in type_counts.items():
            korean_name = self._event_type_to_korean(event_type)
            stats_parts.append(f"{korean_name}: {count}")

        self.stats_label.setText(" | ".join(stats_parts))

    def _show_message(self, message: str):
        self.log_text.setPlainText(message)
