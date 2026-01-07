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
    QAbstractItemView,
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
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._open_context_menu)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

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

    def _on_tree_clicked(self, index):
        file_path = Path(self.model.filePath(index))
        target_dir = None

        if (
            file_path.is_file()
            and file_path.name.startswith("events")
            and file_path.suffix == ".jsonl"
        ):
            target_dir = file_path.parent
        elif file_path.is_dir():
            has_log_file = any(file_path.glob("events*.jsonl"))
            if has_log_file:
                target_dir = file_path

        if target_dir:
            self.log_selected.emit(target_dir)

    def _show_tensorboard(self, index):
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

        tb_action = QAction("열기 (TensorBoard)", self)
        tb_action.triggered.connect(lambda: self._show_tensorboard(index))
        menu.addAction(tb_action)

        menu.addSeparator()

        selection = self.tree.selectionModel().selectedRows(0)

        if index in selection:
            target_indexes = selection
        else:
            target_indexes = [index]

        count = len(target_indexes)
        label = f"삭제 ({count}개 항목)" if count > 1 else "삭제 (Delete)"

        delete_action = QAction(label, self)
        delete_action.triggered.connect(lambda: self._delete_folders(target_indexes))
        menu.addAction(delete_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _delete_folders(self, indexes):
        file_paths = [Path(self.model.filePath(idx)) for idx in indexes]
        paths = list(set(file_paths))
        count = len(paths)
        success_count = 0

        # 삭제 여부 재확인
        reply = QMessageBox.question(
            self,
            "삭제 확인",
            f"정말로 {count}개 항목을 영구 삭제하시겠습니까?\n\n",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        for file_path in paths:
            try:
                # 이미 지워졌는지 확인 (예: 상위 폴더를 지워서 같이 지워진 경우)
                if not file_path.exists():
                    continue

                if file_path.is_dir():
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                success_count += 1

            except Exception as e:
                error_msg += f"\n- {file_path.name}: {e}"
