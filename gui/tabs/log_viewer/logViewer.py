import json
import subprocess  # [추가] 프로세스 실행용
import webbrowser  # [추가] 브라우저 오픈용
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter, QMessageBox
from PyQt6.QtCore import Qt

from core.engine.state import GameEvent
from core.managers.logger import LogManager
from .logLeft import LogLeft
from .logRight import LogRight
from .logEvent import LogEvent


class LogViewer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_log_dir: Optional[Path] = None
        self.tb_process = None  # 텐서보드 변수
        self._setup_ui()

    def _setup_ui(self):
        # 전체 레이아웃 (좌우 분할)
        layout = QHBoxLayout()
        self.setLayout(layout)

        # 1. 좌측 탐색기
        self.explorer = LogLeft()
        self.explorer.log_selected.connect(self._on_log_selected)
        self.explorer.tensorboard_requested.connect(self._launch_tensorboard)

        # 2. 우측 뷰어
        self.content_viewer = LogRight()

        # 3. 스플리터로 결합
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.explorer)
        splitter.addWidget(self.content_viewer)
        splitter.setStretchFactor(1, 1)  # 우측을 더 넓게

        layout.addWidget(splitter)

    def _on_log_selected(self, path: Path):
        """좌측에서 로그 선택 시 호출"""
        self.current_log_dir = path
        self._load_logs(path)

    def _launch_tensorboard(self, tb_path: Path):
        # 초기화
        if self.tb_process:
            try:
                self.tb_process.terminate()
                self.tb_process = None
                print("[GUI] Stopped previous TensorBoard process.")
            except Exception as e:
                print(f"[GUI] Error killing TensorBoard: {e}")

        # 텐서보드 실행
        cmd = ["tensorboard", "--logdir", str(tb_path), "--port", "6006"]

        try:
            self.tb_process = subprocess.Popen(cmd, shell=False)

            print(f"[GUI] Started TensorBoard on port 6006 for {tb_path.name}")

            # 브라우저 열기
            webbrowser.open("http://localhost:6006")

        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "오류",
                "tensorboard 명령을 찾을 수 없습니다.\n환경 변수에 등록되어 있는지 확인하세요.",
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"텐서보드 실행 실패:\n{e}")

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

    def closeEvent(self, event):
        if self.tb_process:
            self.tb_process.terminate()
        super().closeEvent(event)
