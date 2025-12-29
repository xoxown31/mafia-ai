import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# [중요] 새로 만든 calculate_win_stats 함수를 임포트
from utils.log_parser import parse_game_logs, calculate_avg_days, calculate_win_stats


class TeamStatsTab:
    def __init__(self, parent, log_path):
        self.frame = ttk.Frame(parent)
        self.log_path = log_path
        self._init_ui()

    def _init_ui(self):
        top_frame = ttk.Frame(self.frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        btn_refresh = ttk.Button(top_frame, text="새로고침", command=self.refresh)
        btn_refresh.pack(side=tk.RIGHT)

        self.stats_info_frame = ttk.LabelFrame(self.frame, text="AI 역할별 상세 전적")
        self.stats_info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.lbl_ai_mafia = ttk.Label(
            self.stats_info_frame,
            text="마피아: -",
            foreground="red",
            font=("맑은 고딕", 11, "bold"),
        )
        self.lbl_ai_mafia.pack(side=tk.LEFT, padx=20, pady=10)

        self.lbl_ai_citizen = ttk.Label(
            self.stats_info_frame,
            text="시민팀: -",
            foreground="blue",
            font=("맑은 고딕", 11, "bold"),
        )
        self.lbl_ai_citizen.pack(side=tk.LEFT, padx=20, pady=10)

        self.lbl_avg_days = ttk.Label(
            self.stats_info_frame, text="평균 진행: -", font=("맑은 고딕", 11)
        )
        self.lbl_avg_days.pack(side=tk.RIGHT, padx=20, pady=10)

        self.chart_frame = ttk.Frame(self.frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def refresh(self):
        # 1. 기존 그래프 삭제
        for w in self.chart_frame.winfo_children():
            w.destroy()

        # 2. 데이터 가져오기 (파싱)
        games = parse_game_logs(self.log_path)
        if not games:
            ttk.Label(self.chart_frame, text="데이터가 없습니다.").pack()
            return

        # 3. [핵심] 계산 로직은 log_parser에게 맡김
        team_wins, stats = calculate_win_stats(games)

        # 4. 결과 표시
        self._update_labels(stats)
        self._draw_chart(team_wins)

    def _update_labels(self, s):
        # 마피아 전적 표시
        if s["m_play"] > 0:
            rate = (s["m_win"] / s["m_play"]) * 100
            self.lbl_ai_mafia.config(
                text=f"마피아 배정 시: {rate:.1f}% ({s['m_win']}/{s['m_play']}승)"
            )
        else:
            self.lbl_ai_mafia.config(text="마피아 배정 시: 기록 없음")

        # 시민팀 전적 표시
        if s["c_play"] > 0:
            rate = (s["c_win"] / s["c_play"]) * 100
            self.lbl_ai_citizen.config(
                text=f"시민팀 배정 시: {rate:.1f}% ({s['c_win']}/{s['c_play']}승)"
            )
        else:
            self.lbl_ai_citizen.config(text="시민팀 배정 시: 기록 없음")

        # 평균 일수 표시
        avg = calculate_avg_days(self.log_path)
        self.lbl_avg_days.config(text=f"평균 진행: {avg:.1f}일")

    def _draw_chart(self, team_wins):
        total_games = sum(team_wins.values())
        if total_games == 0:
            ttk.Label(self.chart_frame, text="승리 데이터가 충분하지 않습니다.").pack()
            return

        # 캔버스 생성
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        # 데이터 준비
        sizes = [team_wins["CITIZEN"], team_wins["MAFIA"]]
        labels = ["CITIZEN", "MAFIA"]
        colors = ["#8AB4F8", "#174EA6"]

        # [변경점] 도넛 차트 그리기
        # wedgeprops: 도넛의 두께(width)와 테두리 색상(edgecolor) 설정
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 12, "weight": "bold"},
            wedgeprops={"width": 0.4, "edgecolor": "white"},  # 가운데 구멍 뚫기
            pctdistance=0.85,  # % 표시 위치를 도넛 안쪽으로 조정
        )

        # [변경점] 도넛 차트 가운데에 총 게임 수 표시
        ax.text(
            0,
            0,
            f"Total\n{total_games}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#333333",
        )

        ax.set_title(
            "팀별 승리 점유율",
            fontdict={
                "fontsize": 14,
                "fontweight": "bold",
                "fontname": "Malgun Gothic",  # Tkinter 라벨과 동일한 폰트
            },
        )

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        plt.close(fig)
