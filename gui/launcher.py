from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, Qt
from argparse import Namespace


class Launcher(QWidget):
    start_simulation_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI Simulation")
        self.setGeometry(100, 100, 400, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("마피아 AI 시물레이터")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # 에이전트 추가시 수정 부분
        agent_group = QGroupBox("플레이어 에이전트")
        agent_layout = QVBoxLayout()
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(["llm", "ppo", "reinforce"])
        agent_layout.addWidget(self.agent_combo)
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)

        # 2. 실행 모드 선택
        mode_group = QGroupBox("실행 모드")
        mode_layout = QHBoxLayout()
        self.radio_train = QRadioButton("학습 (Train)")
        self.radio_test = QRadioButton("실습/테스트 (Test)")
        self.radio_test.setChecked(True)

        btn_group = QButtonGroup(self)  # 라디오 버튼 그룹핑
        btn_group.addButton(self.radio_train)
        btn_group.addButton(self.radio_test)

        mode_layout.addWidget(self.radio_train)
        mode_layout.addWidget(self.radio_test)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        ep_group = QGroupBox("진행 에피소드 수")
        ep_layout = QVBoxLayout()
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(1, 10000)
        self.ep_spin.setValue(1)
        ep_layout.addWidget(self.ep_spin)
        ep_group.setLayout(ep_layout)
        layout.addWidget(ep_group)

        self.btn_start = QPushButton("시뮬레이션 시작")
        self.btn_start.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        )
        self.btn_start.clicked.connect(self.on_click_start)
        layout.addWidget(self.btn_start)

        self.setLayout(layout)

    def on_click_start(self):
        agent = self.agent_combo.currentText()
        mode = "train" if self.radio_train.isChecked() else "test"  # train ot test

        args = Namespace(
            agent=agent, mode=mode, episodes=self.ep_spin.value(), gui=True
        )

        self.start_simulation_signal.emit(args)
