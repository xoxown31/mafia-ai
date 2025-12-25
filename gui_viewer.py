import sys  # [추가] 프로세스 강제 종료를 위해 필요
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

        # X 버튼(창 닫기)을 누르면 on_closing 함수 실행
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # 종료 처리 함수
    def on_closing(self):
        plt.close("all")
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    # ------------------------------------------------------------------
    # 탭 1: 팀별 승률 (Pie Chart) + AI 역할별 상세 승률
    # ------------------------------------------------------------------
    def _init_team_stats_tab(self):
        frame = ttk.Frame(self.tab_team_stats)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 상단 컨트롤 프레임 (새로고침 버튼 등)
        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        btn_refresh = ttk.Button(
            top_frame, text="새로고침", command=self.refresh_team_stats
        )
        btn_refresh.pack(side=tk.RIGHT)

        # [추가] AI 역할별 승률을 표시할 라벨 프레임
        self.stats_info_frame = ttk.LabelFrame(frame, text="AI 역할별 상세 전적")
        self.stats_info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # 마피아일 때 승률 라벨
        self.lbl_ai_mafia_stats = ttk.Label(
            self.stats_info_frame,
            text="마피아 배정 시: -",
            foreground="red",
            font=("맑은 고딕", 11, "bold"),
        )
        self.lbl_ai_mafia_stats.pack(side=tk.LEFT, padx=20, pady=10)

        # 시민팀일 때 승률 라벨
        self.lbl_ai_citizen_stats = ttk.Label(
            self.stats_info_frame,
            text="시민팀 배정 시: -",
            foreground="blue",
            font=("맑은 고딕", 11, "bold"),
        )
        self.lbl_ai_citizen_stats.pack(side=tk.LEFT, padx=20, pady=10)

        # 평균 진행 일수 라벨 (위치 이동)
        self.lbl_avg_days = ttk.Label(
            self.stats_info_frame, text="평균 진행: -", font=("맑은 고딕", 11)
        )
        self.lbl_avg_days.pack(side=tk.RIGHT, padx=20, pady=10)

        # 차트 프레임
        self.chart_frame_team = ttk.Frame(frame)
        self.chart_frame_team.pack(fill=tk.BOTH, expand=True)

    def refresh_team_stats(self):
        """팀별 승률 및 AI 역할별 상세 승률 갱신"""
        # 기존 차트 지우기
        for widget in self.chart_frame_team.winfo_children():
            widget.destroy()

        from utils.analysis import parse_game_data

        games = parse_game_data(self.log_file_path)

        if not games:
            ttk.Label(self.chart_frame_team, text="데이터가 없습니다.").pack()
            return

        # --- [데이터 집계] ---
        team_wins = {"MAFIA": 0, "CITIZEN": 0}

        # AI 역할별 승률 계산 변수
        ai_mafia_plays = 0
        ai_mafia_wins = 0
        ai_citizen_plays = 0
        ai_citizen_wins = 0

        for game in games:
            winner = game.get("winner")

            # 1. 전체 팀 승률 집계
            if winner == "Mafia":
                team_wins["MAFIA"] += 1
            elif winner == "Citizen":
                team_wins["CITIZEN"] += 1

            # 2. AI(Player 0) 승률 집계
            ai_role = game.get("roles", {}).get(0)
            if ai_role:
                if ai_role == "MAFIA":
                    ai_mafia_plays += 1
                    if winner == "Mafia":
                        ai_mafia_wins += 1
                else:
                    ai_citizen_plays += 1
                    if winner == "Citizen":
                        ai_citizen_wins += 1

        # --- [UI 업데이트] ---

        # (1) 마피아일 때 승률
        if ai_mafia_plays > 0:
            m_rate = (ai_mafia_wins / ai_mafia_plays) * 100
            self.lbl_ai_mafia_stats.config(
                text=f"마피아 배정 시: {m_rate:.1f}% ({ai_mafia_wins}/{ai_mafia_plays}승)"
            )
        else:
            self.lbl_ai_mafia_stats.config(text="마피아 배정 시: 기록 없음")

        # (2) 시민팀일 때 승률
        if ai_citizen_plays > 0:
            c_rate = (ai_citizen_wins / ai_citizen_plays) * 100
            self.lbl_ai_citizen_stats.config(
                text=f"시민팀 배정 시: {c_rate:.1f}% ({ai_citizen_wins}/{ai_citizen_plays}승)"
            )
        else:
            self.lbl_ai_citizen_stats.config(text="시민팀 배정 시: 기록 없음")

        # (3) [수정됨] 평균 진행 일수 (직접 계산 함수 호출)
        avg_days = self._calculate_avg_days_directly()
        self.lbl_avg_days.config(text=f"평균 진행: {avg_days:.1f}일")

        # --- 차트 그리기 ---
        self._draw_team_pie_chart(team_wins)

    def _calculate_avg_days_directly(self):
        """
        로그 파일을 직접 읽어서 에피소드별 진행 일수(Day)의 평균을 계산합니다.
        가정: 로그 내에 'Day 1', 'Day 2' ... 형식의 텍스트가 존재함.
        """
        if not os.path.exists(self.log_file_path):
            return 0.0

        total_days_sum = 0
        episode_count = 0

        current_max_day = 0
        in_episode = False

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # 에피소드 시작 감지
                    if "Episode" in line and "Start" in line:
                        in_episode = True
                        current_max_day = 0  # 일수 초기화

                    # Day 정보 추출 (예: "Day 1 Start", "Day: 2", "Day 3")
                    # 정규식: Day 뒤에 오는 숫자를 찾음
                    if in_episode:
                        day_match = re.search(r"Day\s*(\d+)", line, re.IGNORECASE)
                        if day_match:
                            day_num = int(day_match.group(1))
                            # 현재 에피소드에서 가장 큰 Day 숫자를 유지
                            if day_num > current_max_day:
                                current_max_day = day_num

                    # 에피소드 종료 감지
                    if "Episode" in line and "End" in line:
                        if in_episode:
                            # 0일로 끝나는 경우(시작하자마자 종료) 최소 1일로 보정할지 여부는 선택
                            # 여기서는 발견된 최대 Day를 더함
                            total_days_sum += max(1, current_max_day)
                            episode_count += 1
                        in_episode = False

            if episode_count == 0:
                return 0.0

            return total_days_sum / episode_count

        except Exception as e:
            print(f"일수 계산 중 오류 발생: {e}")
            return 0.0

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

        # [신규 추가] 상단 헤더 프레임 (라벨 + 새로고침 버튼)
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(header_frame, text="에피소드 목록").pack(side=tk.LEFT)

        # 목록 새로고침 버튼 추가
        btn_refresh_list = ttk.Button(
            header_frame, text="새로고침", width=8, command=self.refresh_episode_list
        )
        btn_refresh_list.pack(side=tk.RIGHT)

        # 리스트박스 및 스크롤바
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
