from typing import List, Optional
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QTextEdit,
    QGroupBox,
)

from core.engine.state import GameEvent
from config import Phase, EventType
from core.managers.logger import LogManager


class LogEvent(GameEvent):
    """state.py 수정 없이 episode 필드를 인식하기 위한 확장 클래스"""

    episode: int = 1


class LogRight(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.events: List[LogEvent] = []
        self.log_manager: Optional[LogManager] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        filter_group = QGroupBox("필터")
        filter_layout = QHBoxLayout()
        filter_group.setLayout(filter_layout)

        # Episode
        filter_layout.addWidget(QLabel("Episode:"))
        self.episode_combo = QComboBox()
        self.episode_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.episode_combo)

        # Day
        filter_layout.addWidget(QLabel("Day:"))
        self.day_combo = QComboBox()
        self.day_combo.addItem("전체")
        self.day_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.day_combo)

        # Phase
        filter_layout.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(["전체", "낮 토론", "투표", "처형 여부 결정", "밤"])
        self.phase_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.phase_combo)

        # Type
        filter_layout.addWidget(QLabel("Type:"))
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(
            ["전체", "주장", "투표", "처형", "살해", "보호", "조사"]
        )
        self.event_type_combo.currentTextChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.event_type_combo)

        filter_layout.addStretch()
        layout.addWidget(filter_group)

        # 2. 로그 텍스트
        log_group = QGroupBox("게임 이벤트 로그")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: '맑은 고딕', 'Malgun Gothic'; font-size: 13px;"
        )
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group, stretch=1)

        # 3. 통계
        stats_group = QGroupBox("통계")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        self.stats_label = QLabel("이벤트 로드되지 않음")
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_group)

    def set_data(self, events: List[LogEvent], log_manager: Optional[LogManager]):
        """데이터를 받아서 필터 갱신 및 표시"""
        self.events = events
        self.log_manager = log_manager

        # Episode 필터 갱신
        episodes = sorted(list(set(e.episode for e in self.events)))
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems([str(ep) for ep in episodes])
        self.episode_combo.blockSignals(False)

        # Day 필터 갱신
        days = sorted(set(e.day for e in self.events))
        self.day_combo.blockSignals(True)
        self.day_combo.clear()
        self.day_combo.addItem("전체")
        self.day_combo.addItems([f"Day {d}" for d in days])
        self.day_combo.blockSignals(False)

        self.apply_filter()

    def apply_filter(self):
        if not self.events:
            self.log_text.clear()
            return

        ep_filter = self.episode_combo.currentText()
        day_filter = self.day_combo.currentText()
        phase_filter = self.phase_combo.currentText()
        type_filter = self.event_type_combo.currentText()

        filtered = []
        for event in self.events:
            # Episode
            try:
                if event.episode != int(ep_filter):
                    continue
            except ValueError:
                continue
            # Day
            if day_filter != "전체" and event.day != int(day_filter.split()[1]):
                continue
            # Phase
            if (
                phase_filter != "전체"
                and self._phase_to_korean(event.phase) != phase_filter
            ):
                continue
            # Type
            if (
                type_filter != "전체"
                and self._event_type_to_korean(event.event_type) != type_filter
            ):
                continue

            filtered.append(event)

        self._display_logs(filtered)
        self._update_stats(filtered)

    def _display_logs(self, events: List[LogEvent]):
        self.log_text.clear()
        if not events:
            self.log_text.setPlainText("조건에 맞는 이벤트가 없습니다.")
            return

        grouped = defaultdict(list)
        for event in events:
            grouped[(event.day, event.phase)].append(event)

        html = ""
        for (day, phase), group_events in sorted(grouped.items()):
            phase_str = self._phase_to_korean(phase)
            html += f"<h3 style='color: #0066cc;'>═══ Day {day} - {phase_str} ═══</h3>"
            for event in group_events:
                text = self._format_event(event)
                color = self._get_event_color(event.event_type)
                style = f"color: {color};"
                if event.event_type == EventType.EXECUTE:
                    style += " font-weight: bold;"
                html += f"<div style='margin-left: 10px; {style}'>• {text}</div>"
            html += "<br>"

        self.log_text.setHtml(html)

    def _update_stats(self, events: List[LogEvent]):
        if not events:
            self.stats_label.setText("이벤트 없음")
            return

        counts = defaultdict(int)
        for e in events:
            counts[e.event_type] += 1

        parts = [f"총 이벤트: {len(events)}"]
        for et, cnt in counts.items():
            parts.append(f"{self._event_type_to_korean(et)}: {cnt}")
        self.stats_label.setText(" | ".join(parts))

    # --- Helper Helpers ---
    def _format_event(self, event: GameEvent) -> str:
        if self.log_manager:
            try:
                return self.log_manager.interpret_event(event)
            except:
                pass
        return f"[{event.event_type.name}] Actor: {event.actor_id}, Target: {event.target_id}"

    def _get_event_color(self, et: EventType):
        return {
            EventType.VOTE: "#ff6600",
            EventType.EXECUTE: "#cc0000",
            EventType.KILL: "#FF00F2",
            EventType.PROTECT: "#009900",
            EventType.POLICE_RESULT: "#6ee2ff",
        }.get(et, "#f6f6f8")

    def _phase_to_korean(self, p: Phase):
        return {
            Phase.DAY_DISCUSSION: "낮 토론",
            Phase.DAY_VOTE: "투표",
            Phase.DAY_EXECUTE: "처형 여부 결정",
            Phase.NIGHT: "밤",
        }.get(p, p.name)

    def _event_type_to_korean(self, et: EventType):
        return {
            EventType.CLAIM: "주장",
            EventType.VOTE: "투표",
            EventType.EXECUTE: "처형",
            EventType.KILL: "살해",
            EventType.PROTECT: "보호",
            EventType.POLICE_RESULT: "조사",
        }.get(et, et.name)
