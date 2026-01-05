import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget
from PyQt6.QtGui import QFont
from PyQt6.QtGui import QIcon

from .tabs.log_viewer import LogViewerTab
from pathlib import Path


class MafiaLogViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI 게임 로그 뷰어")
        icon_path = Path(__file__).parent / "icon.jpg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.resize(1100, 750)
        self._load_stylesheet()  # 폰트 설정
        # 중앙 위젯, 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.log_viewer_tab = LogViewerTab(self)
        self.tab_widget.addTab(self.log_viewer_tab, "로그 뷰어")

    def _load_stylesheet(self):
        """styles.qss 파일을 읽어서 적용"""
        try:
            # 현재 파일(launcher.py)과 같은 폴더에 있는 styles.qss 경로 찾기
            qss_path = Path(__file__).parent / "styles.qss"

            if qss_path.exists():
                with open(qss_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            else:
                print(f"Warning: Stylesheet file not found at {qss_path}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    def show_live(self, log_path):
        self.tab_widget.setCurrentWidget(self.log_viewer_tab)
        self.log_viewer_tab.select_live(log_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MafiaLogViewerWindow()
    window.show()
    sys.exit(app.exec())
