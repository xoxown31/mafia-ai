import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSplitter,
)
from PyQt6.QtCore import Qt

from core.engine.state import GameEvent
from core.managers.logger import LogManager
from .logLeft import LogLeft
from .logRight import LogRight
from .logEvent import LogEvent


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
        self.explorer = LogLeft()
        self.explorer.log_selected.connect(self._on_log_selected)

        # 2. 우측 뷰어
        self.content_viewer = LogRight()

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
