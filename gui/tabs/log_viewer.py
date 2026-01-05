import json
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QTreeView,
    QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QDir
from PyQt6.QtGui import QFileSystemModel

from core.engine.state import GameEvent
from config import Role, Phase, EventType
from core.managers.logger import LogManager


class LogEvent(GameEvent):
    """state.py 수정 없이 episode 필드를 인식하기 위한 확장 클래스"""

    episode: int = 1


class LogExplorerWidget(QWidget):
    log_selected = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_path: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # 1. 헤더
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("📂 로그 탐색기"))

        btn_refresh = QPushButton("⟳")
        btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("목록 새로고침")
        btn_refresh.clicked.connect(self._refresh_tree)
        header_layout.addWidget(btn_refresh)
        layout.addLayout(header_layout)

        # 2. 경로 변경 버튼
        self.btn_change_root = QPushButton("다른 폴더 열기...")
        self.btn_change_root.setStyleSheet("font-size: 11px; padding: 3px;")
        self.btn_change_root.clicked.connect(self._change_root_directory)
        layout.addWidget(self.btn_change_root)

        # 3. 모델 설정
        self.model = QFileSystemModel()
        self.model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot
        )
        self.model.setNameFilters(["*.jsonl"])
        self.model.setNameFilterDisables(False)

        # 4. 트리 뷰 설정
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setColumnHidden(1, True)  # Size
        self.tree.setColumnHidden(2, True)  # Type
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )

        self.tree.setStyleSheet(
            """
            QTreeView { border: 1px solid #444; background-color: #222; color: #ddd; }
            QTreeView::item:hover { background-color: #333; }
            QTreeView::item:selected { background-color: #4CAF50; color: white; }
        """
        )
        self.tree.clicked.connect(self._on_tree_clicked)
        layout.addWidget(self.tree)

        # 5. 초기 경로 설정 (자동으로 logs 폴더 잡기)
        self._init_default_logs_path()

    def _init_default_logs_path(self):
        # 현재 파일(gui/tabs/log_viewer.py) 기준 프로젝트 루트 찾기
        project_root = Path(__file__).parent.parent.parent.resolve()
        default_logs = project_root / "logs"

        if not default_logs.exists():
            try:
                default_logs.mkdir(parents=True, exist_ok=True)
            except:
                pass

        self.set_tree_root(default_logs)

    def set_tree_root(self, path: Path):
        if not path.exists():
            return
        self.root_path = path
        self.model.setRootPath(str(path))
        self.tree.setRootIndex(self.model.index(str(path)))

    def _refresh_tree(self):
        if self.root_path:
            self.model.setRootPath(str(self.root_path))

    def _change_root_directory(self):
        start_dir = self.root_path if self.root_path else Path.cwd()
        directory = QFileDialog.getExistingDirectory(
            self, "로그 폴더 선택", str(start_dir)
        )
        if directory:
            self.set_tree_root(Path(directory))

    def _on_tree_clicked(self, index):
        file_path = Path(self.model.filePath(index))
        target_dir = None
        if file_path.is_file() and file_path.name == "events.jsonl":
            target_dir = file_path.parent
        elif file_path.is_dir() and (file_path / "events.jsonl").exists():
            target_dir = file_path

        if target_dir:
            self.log_selected.emit(target_dir)


# === 로그 컨텐츠 뷰어 위젯 ===
class LogContentWidget(QWidget):
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


# 통합 뷰어
class LogViewerTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_log_dir: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        # 전체 레이아웃 (좌우 분할)
        layout = QHBoxLayout()
        self.setLayout(layout)

        # 1. 좌측 탐색기
        self.explorer = LogExplorerWidget()
        self.explorer.log_selected.connect(self._on_log_selected)

        # 2. 우측 뷰어
        self.content_viewer = LogContentWidget()

        # 3. 스플리터로 결합
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.explorer)
        splitter.addWidget(self.content_viewer)
        splitter.setStretchFactor(1, 1)  # 우측을 더 넓게

        layout.addWidget(splitter)

    def select_live(self, base_path_str):
        """라이브 모드 진입 (Launcher에서 호출)"""
        path = Path(base_path_str)
        if path.exists():
            self.explorer.set_root_directory(path)

    def _on_log_selected(self, path: Path):
        """좌측에서 로그 선택 시 호출"""
        self.current_log_dir = path
        self._load_logs(path)

    def _load_logs(self, log_dir: Path):
        """파일 로드 및 파싱 -> ContentWidget으로 전달"""
        jsonl_path = log_dir / "events.jsonl"
        if not jsonl_path.exists():
            self.content_viewer.log_text.setPlainText("로그 파일이 존재하지 않습니다.")
            return

        # LogManager 초기화
        log_manager = None
        try:
            log_manager = LogManager(
                experiment_name="viewer",
                log_dir=str(log_dir.parent),
                use_tensorboard=False,
                write_mode=False,
            )
        except Exception as e:
            print(f"LogManager Init Fail: {e}")

        # 파싱
        events = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = LogEvent(**data)
                        events.append(event)
        except Exception as e:
            self.content_viewer.log_text.setPlainText(f"로그 로드 실패: {e}")
            return

        # 우측 뷰어에 데이터 주입
        self.content_viewer.set_data(events, log_manager)
