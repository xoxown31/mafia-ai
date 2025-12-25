import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import os
from collections import defaultdict

# [설정] 한글 폰트 설정 (깨짐 방지)
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf"
    ).get_name()
    rc("font", family=font_name)
elif platform.system() == "Darwin":
    rc("font", family="AppleGothic")
else:
    rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

# [설정] 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_NAME = "mafia_game_log.txt"


class MafiaLogViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mafia AI 통합 뷰어")
        self.root.geometry("1100x750")

        # 탭 컨트롤 생성
        self.tab_control = ttk.Notebook(root)

        # 탭 1: 팀별 승률 (파이 차트)
        self.tab_team_stats = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_team_stats, text="팀별 승률")
        self._init_team_stats_tab()

        # 탭 2: AI 성장 그래프 (막대 차트) - [신규 추가]
        self.tab_ai_stats = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_ai_stats, text="AI 성장 그래프")
        self._init_ai_stats_tab()

        # 탭 3: 에피소드 리플레이
        self.tab_replay = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_replay, text="에피소드 리플레이")
        self._init_replay_tab()

        self.tab_control.pack(expand=1, fill="both")

        # 데이터 저장소
        self.episode_start_pos = {}
        self.log_file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

        # 시작 시 리플레이 목록 로딩
        self.root.after(100, self.refresh_episode_list)

    # ------------------------------------------------------------------
    # 탭 1: 팀별 승률 (Pie Chart) 관련 함수
    # ------------------------------------------------------------------
    def _init_team_stats_tab(self):
        frame = ttk.Frame(self.tab_team_stats)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_refresh = ttk.Button(
            frame, text="새로고침", command=self.refresh_team_stats
        )
        btn_refresh.pack(side=tk.TOP, anchor=tk.E, pady=(0, 10))

        self.chart_frame_team = ttk.Frame(frame)
        self.chart_frame_team.pack(fill=tk.BOTH, expand=True)

    def refresh_team_stats(self):
        """팀별 승률(파이 차트) 갱신"""
        for widget in self.chart_frame_team.winfo_children():
            widget.destroy()

        from utils.analysis import parse_game_data

        games = parse_game_data(self.log_file_path)

        if not games:
            ttk.Label(self.chart_frame_team, text="데이터가 없습니다.").pack()
            return

        team_wins = {"MAFIA": 0, "CITIZEN": 0}
        total_games = 0

        for game in games:
            winner = game.get("winner")
            if winner == "Mafia":
                team_wins["MAFIA"] += 1
                total_games += 1
            elif winner == "Citizen":
                team_wins["CITIZEN"] += 1
                total_games += 1

        # 차트 그리기
        self._draw_team_pie_chart(team_wins)

    def _draw_team_pie_chart(self, team_wins):
        if sum(team_wins.values()) == 0:
            ttk.Label(self.chart_frame_team, text="승리 데이터가 없습니다.").pack()
            return

        labels = ["CITIZEN", "MAFIA"]
        sizes = [team_wins["CITIZEN"], team_wins["MAFIA"]]
        colors = ["#99ff99", "#ff9999"]  # 초록, 빨강

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 12, "weight": "bold"},
        )
        ax.set_title("팀별 승리 점유율 (전체)")

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_team)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # 탭 2: AI 성장 그래프 (Bar Chart) 관련 함수
    # ------------------------------------------------------------------
    def _init_ai_stats_tab(self):
        frame = ttk.Frame(self.tab_ai_stats)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_refresh = ttk.Button(frame, text="새로고침", command=self.refresh_ai_stats)
        btn_refresh.pack(side=tk.TOP, anchor=tk.E, pady=(0, 10))

        self.chart_frame_ai = ttk.Frame(frame)
        self.chart_frame_ai.pack(fill=tk.BOTH, expand=True)

    def refresh_ai_stats(self):
        """AI 구간별 승률(막대 차트) 갱신"""
        for widget in self.chart_frame_ai.winfo_children():
            widget.destroy()

        from utils.analysis import parse_game_data

        games = parse_game_data(self.log_file_path)

        if not games:
            ttk.Label(self.chart_frame_ai, text="데이터가 없습니다.").pack()
            return

        # 100게임 단위 구간 계산
        CHUNK_SIZE = 100
        total_games = len(games)
        x_labels = []
        y_values = []

        for i in range(0, total_games, CHUNK_SIZE):
            chunk = games[i : i + CHUNK_SIZE]
            if not chunk:
                continue

            wins = 0
            valid_games = 0

            for game in chunk:
                winner = game.get("winner")
                my_role = game["roles"].get(0)  # Player 0 (AI)

                if not winner or not my_role:
                    continue
                valid_games += 1

                # AI 승리 판별
                is_mafia_side = my_role == "MAFIA"
                if (is_mafia_side and winner == "Mafia") or (
                    not is_mafia_side and winner == "Citizen"
                ):
                    wins += 1

            win_rate = (wins / valid_games * 100) if valid_games > 0 else 0

            start_ep = i + 1
            end_ep = min(i + CHUNK_SIZE, total_games)
            x_labels.append(f"{start_ep}\n~\n{end_ep}")
            y_values.append(win_rate)

        # 차트 그리기
        self._draw_ai_bar_chart(x_labels, y_values)

    def _draw_ai_bar_chart(self, x_labels, y_values):
        if not x_labels:
            ttk.Label(self.chart_frame_ai, text="데이터가 부족합니다.").pack()
            return

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        bars = ax.bar(x_labels, y_values, color="#66b3ff", edgecolor="white")

        ax.set_ylim(0, 100)
        ax.set_title("구간별 AI(Player 0) 승률 변화")
        ax.set_ylabel("승률 (%)")
        ax.set_xlabel("에피소드 구간")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.xticks(fontsize=9)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame_ai)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # 탭 3: 리플레이 관련 함수
    # ------------------------------------------------------------------
    def _init_replay_tab(self):
        paned = ttk.PanedWindow(self.tab_replay, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned, width=250)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="에피소드 목록").pack(pady=5)
        list_scroll = ttk.Scrollbar(left_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.episode_listbox = tk.Listbox(
            left_frame, yscrollcommand=list_scroll.set, font=("Consolas", 11)
        )
        self.episode_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.episode_listbox.yview)

        self.episode_listbox.bind("<<ListboxSelect>>", self.on_episode_select)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=4)

        ttk.Label(right_frame, text="게임 진행 로그").pack(pady=5)
        self.log_text = scrolledtext.ScrolledText(
            right_frame, font=("맑은 고딕", 10), state="disabled"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def refresh_episode_list(self):
        self.episode_start_pos = {}
        self.episode_listbox.delete(0, tk.END)

        if not os.path.exists(self.log_file_path):
            return

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break

                    match_start = re.search(r"Episode (\d+) Start", line)
                    if match_start:
                        ep_num = int(match_start.group(1))
                        self.episode_start_pos[ep_num] = pos
                        self.episode_listbox.insert(tk.END, f"Episode {ep_num}")
        except Exception as e:
            messagebox.showerror("오류", f"인덱싱 오류: {e}")

    def on_episode_select(self, event):
        selection = self.episode_listbox.curselection()
        if not selection:
            return

        text = self.episode_listbox.get(selection[0])
        ep_num = int(text.split()[-1])
        start_pos = self.episode_start_pos.get(ep_num)

        content_lines = []
        end_marker = f"Episode {ep_num} End"

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                f.seek(start_pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    content_lines.append(line)
                    if end_marker in line:
                        break

            self.log_text.config(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "".join(content_lines))
            self.log_text.see(1.0)
            self.log_text.config(state="disabled")
        except Exception as e:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MafiaLogViewerApp(root)
    root.mainloop()
