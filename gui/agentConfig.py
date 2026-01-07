from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal


class AgentConfigWidget(QGroupBox):
    """각 플레이어(0~7)를 개별 설정하는 위젯"""

    typeChanged = pyqtSignal()

    def __init__(self, player_id):
        super().__init__(f"Player {player_id}")
        self.player_id = player_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 1. 상단 공통 설정 (Type 및 Role)
        top_layout = QHBoxLayout()

        # [Type 설정]
        top_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LLM", "RL"])
        self.type_combo.setSizePolicy(
            self.type_combo.sizePolicy().horizontalPolicy(),
            self.type_combo.sizePolicy().verticalPolicy(),
        )
        top_layout.addWidget(self.type_combo, stretch=1)

        # [Role 설정] - 공통 영역으로 이동됨
        top_layout.addWidget(QLabel("Role:"))
        self.role_combo = QComboBox()
        # Random을 기본값으로 사용하기 위해 맨 앞에 추가
        self.role_combo.addItems(["Random", "Citizen", "Police", "Doctor", "Mafia"])
        top_layout.addWidget(self.role_combo, stretch=1)

        self.layout.addLayout(top_layout)

        # 2. RL 전용 설정 영역 (RL 선택 시에만 보임)
        self.rl_config_area = QWidget()
        rl_layout = QVBoxLayout()
        self.rl_config_area.setLayout(rl_layout)
        rl_layout.setContentsMargins(0, 0, 0, 0)  # 내부 여백 제거

        # [모델 불러오기 설정]
        model_load_layout = QHBoxLayout()
        rl_layout.addWidget(QLabel("Load Model:"))

        self.load_model_path_input = QLineEdit()
        self.load_model_path_input.setPlaceholderText("선택 안 함 (처음부터 학습)")
        self.load_model_path_input.setReadOnly(True)
        model_load_layout.addWidget(self.load_model_path_input)

        self.btn_select_model = QPushButton("📂")
        self.btn_select_model.setFixedWidth(30)
        self.btn_select_model.clicked.connect(self._select_model_file)
        model_load_layout.addWidget(self.btn_select_model)

        rl_layout.addLayout(model_load_layout)

        # [알고리즘 선택]
        rl_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PPO", "REINFORCE"])
        rl_layout.addWidget(self.algo_combo)

        # [백본 선택]
        rl_layout.addWidget(QLabel("Backbone:"))
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["LSTM", "GRU"])
        rl_layout.addWidget(self.backbone_combo)

        # [은닉층 차원]
        rl_layout.addWidget(QLabel("Hidden Dim:"))
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 512)
        self.hidden_dim_spin.setValue(128)
        rl_layout.addWidget(self.hidden_dim_spin)

        # [RNN 레이어 수]
        rl_layout.addWidget(QLabel("RNN Layers:"))
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 4)
        self.num_layers_spin.setValue(2)
        rl_layout.addWidget(self.num_layers_spin)

        self.layout.addWidget(self.rl_config_area)

        # 초기 상태 설정: 타입에 따라 RL 영역 표시 여부 결정
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self._toggle_rl_area(self.type_combo.currentText())

        self.layout.addStretch()

    def _select_model_file(self):
        """모델 파일(.pt) 선택 다이얼로그 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "학습된 모델 파일 선택",
            "./models",  # 기본 시작 경로
            "Model Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.load_model_path_input.setText(file_path)

    def _on_type_changed(self, text):
        self._toggle_rl_area(text)
        self.typeChanged.emit()

    def _toggle_rl_area(self, agent_type):
        """에이전트 타입에 따라 RL 설정 영역 표시/숨김"""
        self.rl_config_area.setVisible(agent_type == "RL")

    def get_config(self):
        """현재 설정된 에이전트 정보를 딕셔너리로 반환"""
        config = {"type": self.type_combo.currentText().lower()}

        # [수정] 역할(Role) 정보 포함 (RL/LLM 공통)
        config["role"] = self.role_combo.currentText().lower()

        if config["type"] == "rl":
            config["algo"] = self.algo_combo.currentText().lower()
            config["backbone"] = self.backbone_combo.currentText().lower()
            config["hidden_dim"] = self.hidden_dim_spin.value()
            config["num_layers"] = self.num_layers_spin.value()

            # 모델 로드 경로 포함
            path_text = self.load_model_path_input.text().strip()
            config["load_model_path"] = path_text if path_text else None

        return config

    def set_config(
        self,
        agent_type="LLM",
        role="Random",  # [추가] Role 설정 인자
        algo="PPO",
        backbone="LSTM",
        hidden_dim=128,
        num_layers=2,
        load_model_path=None,  # [추가] 모델 경로 인자
    ):
        """외부에서 설정을 일괄 적용할 때 사용"""
        self.type_combo.setCurrentText(agent_type.upper())

        # [추가] Role 설정 반영
        role_text = role.capitalize()
        if self.role_combo.findText(role_text) >= 0:
            self.role_combo.setCurrentText(role_text)
        else:
            self.role_combo.setCurrentText("Random")

        if agent_type.upper() == "RL":
            self.algo_combo.setCurrentText(algo.upper())
            self.backbone_combo.setCurrentText(backbone.upper())
            self.hidden_dim_spin.setValue(hidden_dim)
            self.num_layers_spin.setValue(num_layers)

            if load_model_path:
                self.load_model_path_input.setText(load_model_path)
