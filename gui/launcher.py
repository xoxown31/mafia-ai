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
    QScrollArea,
)
from PyQt6.QtCore import pyqtSignal, Qt
from argparse import Namespace
from pathlib import Path


class AgentConfigWidget(QGroupBox):
    """각 플레이어(0~7)를 개별 설정하는 위젯"""
    
    typeChanged = pyqtSignal()

    def __init__(self, player_id):
        super().__init__(f"Player {player_id}")
        self.player_id = player_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Type:"))

        # 1. 에이전트 메인 타입 (LLM vs RL)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LLM", "RL"])
        self.type_combo.setSizePolicy(
            self.type_combo.sizePolicy().horizontalPolicy(),
            self.type_combo.sizePolicy().verticalPolicy(),
        )
        top_layout.addWidget(self.type_combo, stretch=1)

        self.layout.addLayout(top_layout)

        # 2. RL 전용 설정 영역 (RL 선택 시만 노출/활성화)
        self.rl_config_area = QWidget()
        rl_layout = QVBoxLayout()
        self.rl_config_area.setLayout(rl_layout)
        rl_layout.setContentsMargins(0, 0, 0, 0)  # 내부 여백 제거

        # 알고리즘 선택
        rl_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PPO", "REINFORCE"])
        rl_layout.addWidget(self.algo_combo)
        
        # 백본 선택
        rl_layout.addWidget(QLabel("Backbone:"))
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["LSTM", "GRU"])
        rl_layout.addWidget(self.backbone_combo)
        
        # 은닉층 차원
        rl_layout.addWidget(QLabel("Hidden Dim:"))
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 512)
        self.hidden_dim_spin.setValue(128)
        rl_layout.addWidget(self.hidden_dim_spin)

        # RNN 레이어 수 (LSTM/GRU용)
        rl_layout.addWidget(QLabel("RNN Layers:"))
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 4)
        self.num_layers_spin.setValue(2)
        rl_layout.addWidget(self.num_layers_spin)

        self.layout.addWidget(self.rl_config_area)

        # 타입 변경 시 RL 설정 영역 토글
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self._toggle_rl_area(self.type_combo.currentText())

        self.layout.addStretch()
    
    def _on_type_changed(self, text):
        self._toggle_rl_area(text)
        self.typeChanged.emit()

    def _toggle_rl_area(self, agent_type):
        """에이전트 타입에 따라 RL 설정 영역 표시/숨김"""
        self.rl_config_area.setVisible(agent_type == "RL")

    def get_config(self):
        """현재 설정된 에이전트 정보를 딕셔너리로 반환"""
        config = {"type": self.type_combo.currentText().lower()}
        if config["type"] == "rl":
            config["algo"] = self.algo_combo.currentText().lower()
            config["backbone"] = self.backbone_combo.currentText().lower()
            config["hidden_dim"] = self.hidden_dim_spin.value()
            config["num_layers"] = self.num_layers_spin.value()
        return config
    
    def set_config(self, agent_type="LLM", algo="PPO", backbone="LSTM", hidden_dim=128, num_layers=2):
        """외부에서 설정을 일괄 적용할 때 사용"""
        self.type_combo.setCurrentText(agent_type.upper())
        if agent_type.upper() == "RL":
            self.algo_combo.setCurrentText(algo.upper())
            self.hidden_dim_spin.setValue(hidden_dim)
            self.num_layers_spin.setValue(num_layers)


class Launcher(QWidget):
    start_simulation_signal = pyqtSignal(object)
    stop_simulation_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI Simulation")
        self.resize(450, 600)

        # 8개의 개별 에이전트 설정 위젯을 저장
        self.agent_config_widgets = []

        self._init_ui()

    def _init_ui(self):
        self._load_stylesheet()

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        self.setLayout(self.main_layout)

        self.left_widget = QWidget()
        self.left_widget.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        self.left_widget.setLayout(layout)

        title = QLabel("마피아 AI 시뮬레이터")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #4CAF50; margin-bottom: 10px;"
        )
        layout.addWidget(title)

        # 1. 실행 모드
        self.mode_group = QGroupBox("실행 모드")
        mode_layout = QHBoxLayout()
        self.radio_train = QRadioButton("학습 (Train)")
        self.radio_test = QRadioButton("평가 (Test)")
        self.radio_test.setChecked(True)

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.radio_train)
        btn_group.addButton(self.radio_test)

        mode_layout.addWidget(self.radio_train)
        mode_layout.addWidget(self.radio_test)
        self.mode_group.setLayout(mode_layout)
        layout.addWidget(self.mode_group)

        # 2. 에피소드 수
        ep_group = QGroupBox("진행 에피소드 수")
        ep_layout = QVBoxLayout()
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(1, 10000)
        self.ep_spin.setValue(1)
        ep_layout.addWidget(self.ep_spin)
        ep_group.setLayout(ep_layout)
        layout.addWidget(ep_group)

        # 3. 빠른 설정 (일괄 적용)
        quick_group = QGroupBox("빠른 설정")
        quick_layout = QVBoxLayout()

        quick_desc = QLabel("모든 플레이어에게 동일한 설정 일괄 적용")
        quick_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        quick_layout.addWidget(quick_desc)

        quick_controls = QHBoxLayout()

        self.quick_type_combo = QComboBox()
        self.quick_type_combo.addItems(["LLM", "RL"])
        quick_controls.addWidget(QLabel("Type:"))
        quick_controls.addWidget(self.quick_type_combo)

        btn_apply_all = QPushButton("모두 적용")
        btn_apply_all.clicked.connect(self.apply_to_all_agents)
        quick_controls.addWidget(btn_apply_all)

        quick_layout.addLayout(quick_controls)
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)

        # 4. 경로 관리
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

        # 로그 뷰어 버튼
        self.btn_log_viewer = QPushButton("📊 게임 로그 뷰어 열기")
        self.btn_log_viewer.clicked.connect(self.open_log_viewer)
        layout.addWidget(self.btn_log_viewer)

        # 에이전트 설정 버튼
        self.btn_expand = QPushButton("⚙️ 개별 에이전트 상세 설정")
        self.btn_expand.setCheckable(True)
        self.btn_expand.setToolTip("8명의 에이전트를 개별적으로 설정합니다")
        self.btn_expand.clicked.connect(self.toggle_right_panel)
        layout.addWidget(self.btn_expand)

        # 시작 버튼
        self.btn_start = QPushButton("시뮬레이션 시작")
        self.btn_start.clicked.connect(self.on_click_start)
        layout.addWidget(self.btn_start)

        # 중지 버튼
        self.btn_stop = QPushButton("중지")
        self.btn_stop.clicked.connect(self.on_click_stop)
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        self.right_panel = QGroupBox("개별 에이전트 설정 (8명)")
        self.right_panel.setVisible(False)

        # 그룹박스 메인 레이아웃
        panel_layout = QVBoxLayout()
        self.right_panel.setLayout(panel_layout)

        # 스크롤 영역 생성
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # 스크롤 내부 컨텐츠 위젯
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_layout = QGridLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # 8개의 AgentConfigWidget 생성 및 스크롤 영역에 추가
        for i in range(8):
            agent_widget = AgentConfigWidget(i)
            agent_widget.typeChanged.connect(self.update_mode_visibility)
            self.agent_config_widgets.append(agent_widget)

            row = i // 2
            col = i % 2
            scroll_layout.addWidget(agent_widget, row, col)

        scroll.setWidget(scroll_content)
        panel_layout.addWidget(scroll)

        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_panel)
        
        # 초기 상태 업데이트
        self.update_mode_visibility()

    def toggle_right_panel(self):
        """설정 버튼 클릭 시 패널 열기/닫기"""
        if self.btn_expand.isChecked():
            self.right_panel.setVisible(True)
            self.resize(1150, 750)  # 패널 열릴 때 크기
        else:
            self.right_panel.setVisible(False)
            self.resize(450, 600)  # 패널 닫힐 때 크기
            self.adjustSize()
    
    def update_mode_visibility(self):
        """RL 에이전트 존재 여부에 따라 실행 모드 박스 표시/숨김"""
        has_rl_agent = False
        for widget in self.agent_config_widgets:
            if widget.get_config()["type"] == "rl":
                has_rl_agent = True
                break
        
        self.mode_group.setVisible(has_rl_agent)
        
        # RL 에이전트가 없으면 강제로 Test 모드로 전환
        if not has_rl_agent:
            self.radio_test.setChecked(True)

    def apply_to_all_agents(self):
        """빠른 설정을 모든 에이전트에 일괄 적용"""
        agent_type = self.quick_type_combo.currentText()

        for widget in self.agent_config_widgets:
            widget.set_config(agent_type=agent_type)
        
        self.update_mode_visibility()
        
        QMessageBox.information(
            self, "설정 적용 완료", f"모든 플레이어를 {agent_type}로 설정했습니다."
        )

    def select_model_path(self):
        """모델 저장 경로 선택"""
        path = QFileDialog.getExistingDirectory(
            self, "모델 저장 경로 선택", self.model_path_input.text()
        )
        if path:
            self.model_path_input.setText(path)

    def select_log_path(self):
        """로그 출력 경로 선택"""
        path = QFileDialog.getExistingDirectory(
            self, "로그 출력 경로 선택", self.log_path_input.text()
        )
        if path:
            self.log_path_input.setText(path)

    def open_log_viewer(self):
        """로그 뷰어 창 열기 (PyQt6 윈도우)"""
        # 이전에 만든 gui_viewer.py의 클래스를 import
        try:
            from gui.gui_viewer import MafiaLogViewerWindow

            self.log_window = MafiaLogViewerWindow()
            self.log_window.show()
        except ImportError:
            QMessageBox.warning(
                self, "오류", "gui/gui_viewer.py 파일을 찾을 수 없습니다."
            )

    def on_click_start(self):
        """시뮬레이션 시작 버튼 클릭 - 개별 에이전트 설정 수집"""

        # 8개 에이전트의 개별 설정 수집
        player_configs = [widget.get_config() for widget in self.agent_config_widgets]

        mode = "train" if self.radio_train.isChecked() else "test"

        # 경로 설정
        paths = {
            "model_dir": Path(self.model_path_input.text()),
            "log_dir": Path(self.log_path_input.text()),
        }

        args = Namespace(
            player_configs=player_configs,
            mode=mode,
            episodes=self.ep_spin.value(),
            gui=True,
            paths=paths,
        )
        self.set_btn(False)
        self.start_simulation_signal.emit(args)

    def on_click_stop(self):
        self.set_btn(True)
        self.stop_simulation_signal.emit()

    # 추후에 시뮬레이션 종료시 버튼 복구 기능 추가 구현
    def set_btn(self, run):
        self.btn_start.setEnabled(run)
        self.btn_stop.setEnabled(not run)

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
