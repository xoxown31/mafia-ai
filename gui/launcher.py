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
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from argparse import Namespace
from pathlib import Path


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
        
        # 4. RL 상세 설정
        rl_group = QGroupBox("RL 상세 설정 (PPO/REINFORCE)")
        rl_layout = QGridLayout()
        
        # 알고리즘 선택
        rl_layout.addWidget(QLabel("알고리즘:"), 0, 0)
        self.rl_algorithm = QComboBox()
        self.rl_algorithm.addItems(["PPO", "REINFORCE"])
        rl_layout.addWidget(self.rl_algorithm, 0, 1)
        
        # 백본 선택
        rl_layout.addWidget(QLabel("백본:"), 1, 0)
        self.rl_backbone = QComboBox()
        self.rl_backbone.addItems(["MLP", "LSTM", "GRU"])
        rl_layout.addWidget(self.rl_backbone, 1, 1)
        
        # 은닉층 차원
        rl_layout.addWidget(QLabel("은닉층 차원:"), 2, 0)
        self.rl_hidden_dim = QSpinBox()
        self.rl_hidden_dim.setRange(32, 512)
        self.rl_hidden_dim.setValue(128)
        rl_layout.addWidget(self.rl_hidden_dim, 2, 1)
        
        # RNN 레이어 수
        rl_layout.addWidget(QLabel("RNN 레이어:"), 3, 0)
        self.rl_num_layers = QSpinBox()
        self.rl_num_layers.setRange(1, 4)
        self.rl_num_layers.setValue(2)
        rl_layout.addWidget(self.rl_num_layers, 3, 1)
        
        rl_group.setLayout(rl_layout)
        layout.addWidget(rl_group)
        
        # 5. 경로 관리
        path_group = QGroupBox("경로 관리")
        path_layout = QGridLayout()
        
        # 모델 저장 경로
        path_layout.addWidget(QLabel("모델 저장:"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText("./models")
        self.model_path_input.setReadOnly(True)
        path_layout.addWidget(self.model_path_input, 0, 1)
        
        btn_model_path = QPushButton("📁")
        btn_model_path.setFixedSize(30, 30)
        btn_model_path.clicked.connect(self.select_model_path)
        path_layout.addWidget(btn_model_path, 0, 2)
        
        # 로그 출력 경로
        path_layout.addWidget(QLabel("로그 출력:"), 1, 0)
        self.log_path_input = QLineEdit()
        self.log_path_input.setText("./logs")
        self.log_path_input.setReadOnly(True)
        path_layout.addWidget(self.log_path_input, 1, 1)
        
        btn_log_path = QPushButton("📁")
        btn_log_path.setFixedSize(30, 30)
        btn_log_path.clicked.connect(self.select_log_path)
        path_layout.addWidget(btn_log_path, 1, 2)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        layout.addStretch()

        # 시작 버튼
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
            self.resize(900, 600)
        else:
            self.right_panel.setVisible(False)
            self.resize(400, 550)
            self.adjustSize()
    
    def select_model_path(self):
        """모델 저장 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "모델 저장 경로 선택", self.model_path_input.text())
        if path:
            self.model_path_input.setText(path)
    
    def select_log_path(self):
        """로그 출력 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "로그 출력 경로 선택", self.log_path_input.text())
        if path:
            self.log_path_input.setText(path)

    def sync_sub_agents(self, text):
        """
        메인(플레이어) 에이전트가 변경되면
        오른쪽 8개 박스도 동일한 값으로 변경함
        """
        for combo in self.sub_agent_combos:
            combo.setCurrentText(text)

    def on_click_start(self):
        """시뮬레이션 시작 버튼 클릭 - RL 설정 및 경로 포함"""
        # 메인 에이전트 설정
        main_agent = self.agent_combo.currentText()

        # 오른쪽 8명 에이전트 설정값 수집
        others_agents = [combo.currentText() for combo in self.sub_agent_combos]

        mode = "train" if self.radio_train.isChecked() else "test"

        # RL 상세 설정 수집
        rl_config = {
            "algorithm": self.rl_algorithm.currentText().lower(),
            "backbone": self.rl_backbone.currentText().lower(),
            "hidden_dim": self.rl_hidden_dim.value(),
            "num_layers": self.rl_num_layers.value(),
        }
        
        # 경로 설정
        paths = {
            "model_dir": Path(self.model_path_input.text()),
            "log_dir": Path(self.log_path_input.text()),
        }

        args = Namespace(
            agent=main_agent,
            others=others_agents,
            mode=mode,
            episodes=self.ep_spin.value(),
            gui=True,
            rl_config=rl_config,
            paths=paths,
        )

        self.start_simulation_signal.emit(args)
