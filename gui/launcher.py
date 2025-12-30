from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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
        self.resize(400, 450)

        # 에이전트 타입 목록 정의 (일관성을 위해 리스트로 관리)
        self.agent_types = ["llm", "ppo", "reinforce"]

        # 오른쪽 8개의 콤보박스를 제어하기 위해 리스트에 저장해둠
        self.sub_agent_combos = []

        self._init_ui()

    def _init_ui(self):
        # === [메인 레이아웃] ===
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # =================================================
        # [왼쪽 패널]
        # =================================================
        self.left_widget = QWidget()
        layout = QVBoxLayout()
        self.left_widget.setLayout(layout)

        title = QLabel("마피아 AI 시물레이터")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # 1. 플레이어 에이전트 설정
        agent_group = QGroupBox("플레이어 에이전트 (Main)")
        agent_layout = QHBoxLayout()

        self.agent_combo = QComboBox()
        self.agent_combo.addItems(self.agent_types)  # 정의한 리스트 사용
        # ★ 핵심: 메인 콤보박스가 바뀌면 sync_sub_agents 함수 실행
        self.agent_combo.currentTextChanged.connect(self.sync_sub_agents)
        agent_layout.addWidget(self.agent_combo)

        # 확장 버튼
        self.btn_expand = QPushButton("⚙️")
        self.btn_expand.setFixedSize(30, 30)
        self.btn_expand.setCheckable(True)
        self.btn_expand.setToolTip("에이전트 설정")
        self.btn_expand.clicked.connect(self.toggle_right_panel)
        agent_layout.addWidget(self.btn_expand)

        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)

        # 2. 실행 모드
        mode_group = QGroupBox("실행 모드")
        mode_layout = QHBoxLayout()
        self.radio_train = QRadioButton("학습 (Train)")
        self.radio_test = QRadioButton("평가 (Test)")
        self.radio_test.setChecked(True)

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.radio_train)
        btn_group.addButton(self.radio_test)

        mode_layout.addWidget(self.radio_train)
        mode_layout.addWidget(self.radio_test)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 3. 에피소드 수
        ep_group = QGroupBox("진행 에피소드 수")
        ep_layout = QVBoxLayout()
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(1, 10000)
        self.ep_spin.setValue(1)
        ep_layout.addWidget(self.ep_spin)
        ep_group.setLayout(ep_layout)
        layout.addWidget(ep_group)

        layout.addStretch()

        # 4. 시작 버튼
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

        # 모델 설정
        self.right_panel = QGroupBox("모델 설정")
        self.right_panel.setVisible(False)

        right_layout = QGridLayout()
        self.right_panel.setLayout(right_layout)

        for i in range(8):
            box = QGroupBox(f"Agent {i+1}")
            box_layout = QVBoxLayout()

            # 여기서도 같은 에이전트 타입 목록을 사용
            combo = QComboBox()
            combo.addItems(self.agent_types)

            box_layout.addWidget(combo)
            box.setLayout(box_layout)
            self.sub_agent_combos.append(combo)

            row = i // 2
            col = i % 2
            right_layout.addWidget(box, row, col)

        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_panel)

        self.sync_sub_agents(self.agent_combo.currentText())

    def toggle_right_panel(self):
        """설정 버튼 클릭 시 패널 열기/닫기"""
        if self.btn_expand.isChecked():
            self.right_panel.setVisible(True)
            self.resize(900, 500)
        else:
            self.right_panel.setVisible(False)
            self.resize(400, 450)
            self.adjustSize()

    def sync_sub_agents(self, text):
        """
        메인(플레이어) 에이전트가 변경되면
        오른쪽 8개 박스도 동일한 값으로 변경함
        """
        for combo in self.sub_agent_combos:
            combo.setCurrentText(text)

    def on_click_start(self):
        # 메인 에이전트 설정
        main_agent = self.agent_combo.currentText()

        # 오른쪽 8명 에이전트 설정값 수집 (리스트 형태)
        others_agents = [combo.currentText() for combo in self.sub_agent_combos]

        mode = "train" if self.radio_train.isChecked() else "test"

        args = Namespace(
            agent=main_agent,  # 플레이어(Main)
            others=others_agents,  # 나머지 8명 리스트
            mode=mode,
            episodes=self.ep_spin.value(),
            gui=True,
        )

        self.start_simulation_signal.emit(args)
