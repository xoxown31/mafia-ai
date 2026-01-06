import shutil
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTreeView,
    QHeaderView,
    QMenu,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, QDir, Qt
from PyQt6.QtGui import QFileSystemModel, QAction


class LogLeft(QWidget):
    log_selected = pyqtSignal(Path)
    tensorboard_requested = pyqtSignal(Path)

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
        self.tree.doubleClicked.connect(self._on_tree_double_clicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._open_context_menu)

        layout.addWidget(self.tree)

        # 5. 초기 경로 설정 (자동으로 logs 폴더 잡기)
        self._init_default_logs_path()

    def _init_default_logs_path(self):
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
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

    def _on_tree_double_clicked(self, index):
        file_path = Path(self.model.filePath(index))

        tb_dir = file_path

        if file_path.is_dir() and tb_dir.exists():
            self.tensorboard_requested.emit(tb_dir)
            print(f"[GUI] TensorBoard requested for: {tb_dir}")
        else:
            print(f"no tensorboard in folder (경로: {tb_dir})")

    def _open_context_menu(self, position):
        index = self.tree.indexAt(position)
        if not index.isValid():
            return

        menu = QMenu()

        delete_action = QAction("삭제", self)
        delete_action.triggered.connect(lambda: self._delete_folder(index))
        menu.addAction(delete_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _delete_folder(self, index):
        file_path = Path(self.model.filePath(index))

        # 삭제 여부 재확인
        reply = QMessageBox.question(
            self,
            "삭제 확인",
            f"정말로 다음 항목을 영구 삭제하시겠습니까?\n\n{file_path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 삭제
        try:
            if file_path.is_dir():
                shutil.rmtree(file_path)  # 폴더 삭제 (내부 파일 포함)
            else:
                os.remove(file_path)

            print(f"[GUI] Deleted: {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "삭제 실패", f"삭제 중 오류가 발생했습니다:\n{e}"
            )
