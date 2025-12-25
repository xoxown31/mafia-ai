import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import os
from collections import defaultdict
from utils.analysis import parse_game_data

# [설정] 현재 파일 위치 기준으로 logs 폴더 찾기 (절대 경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_NAME = "mafia_game_log.txt"


class MafiaLogViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mafia AI 로그 뷰어")
        self.root.geometry("1100x750")

        # 탭 컨트롤
        self.tab_control = ttk.Notebook(root)

        # 탭 1: 통계
        self.tab_stats = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_stats, text="승률 통계")
        self._init_stats_tab()

        # 탭 2: 리플레이
        self.tab_replay = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_replay, text="에피소드 리플레이")
        self._init_replay_tab()

        self.tab_control.pack(expand=1, fill="both")

        # 데이터 저장소: 시작 위치만 저장 { 에피소드번호: 시작byte }
        self.episode_start_pos = {}
        self.log_file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

        # 시작하자마자 파일 인덱싱 시작
        self.root.after(100, self.refresh_episode_list)

    def _init_stats_tab(self):
        frame = ttk.Frame(self.tab_stats)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_refresh = ttk.Button(frame, text="새로고침", command=self.refresh_stats)
        btn_refresh.pack(side=tk.TOP, anchor=tk.E, pady=(0, 10))

        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def _init_replay_tab(self):
        paned = ttk.PanedWindow(self.tab_replay, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 좌측: 목록 패널
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

        # 우측: 텍스트 뷰어 패널
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=4)

        ttk.Label(right_frame, text="게임 진행 로그").pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(
            right_frame, font=("맑은 고딕", 10), state="disabled"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def refresh_stats(self):
        """analysis.py의 로직을 재사용하여 통계 갱신"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # [핵심] 직접 파싱하지 않고 analysis.py의 함수 호출!
        games = parse_game_data(self.log_file_path)

        if not games:
            ttk.Label(self.chart_frame, text="데이터가 없습니다.").pack()
            return

        # 데이터를 뷰어 형식에 맞게 집계 (직업별 승률)
        stats = defaultdict(lambda: {"wins": 0, "total": 0})

        for game in games:
            winner = game["winner"]
            if not winner:
                continue

            for pid, role in game["roles"].items():
                stats[role]["total"] += 1

                is_mafia_role = role == "MAFIA"
                if (winner == "Mafia" and is_mafia_role) or (
                    winner == "Citizen" and not is_mafia_role
                ):
                    stats[role]["wins"] += 1

        # 그래프 그리기
        self._draw_chart(stats)

    def _draw_chart(self, stats):
        """통계 데이터를 바탕으로 그래프를 그리는 함수"""
        if not stats:
            ttk.Label(self.chart_frame, text="데이터가 없습니다.").pack()
            return

        roles = sorted(stats.keys())
        # 승률 계산
        win_rates = [
            (stats[r]["wins"] / stats[r]["total"] * 100) if stats[r]["total"] > 0 else 0
            for r in roles
        ]

        # 색상 설정
        color_map = {
            "MAFIA": "#ff9999",
            "CITIZEN": "#99ff99",
            "POLICE": "#66b3ff",
            "DOCTOR": "#ffcc99",
        }
        colors = [color_map.get(r, "#cccccc") for r in roles]

        # Matplotlib 그래프 생성
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        bars = ax.bar(roles, win_rates, color=colors)

        ax.set_ylim(0, 100)
        ax.set_title("직업별 승률")
        ax.set_ylabel("승률 (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # 막대 위에 수치 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        # Tkinter 창에 그래프 삽입
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def refresh_episode_list(self):
        """
        [수정됨] 에피소드의 '시작 위치'만 빠르게 스캔하여 저장
        """
        self.episode_start_pos = {}
        self.episode_listbox.delete(0, tk.END)

        if not os.path.exists(self.log_file_path):
            messagebox.showinfo(
                "알림", f"로그 파일을 찾을 수 없습니다.\n경로: {self.log_file_path}"
            )
            return

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                while True:
                    pos = f.tell()  # 현재 줄의 시작 위치 기억
                    line = f.readline()
                    if not line:
                        break

                    # "Episode N Start" 문구를 찾으면 위치 저장
                    match_start = re.search(r"Episode (\d+) Start", line)
                    if match_start:
                        ep_num = int(match_start.group(1))
                        self.episode_start_pos[ep_num] = pos  # 시작 바이트 저장
                        self.episode_listbox.insert(tk.END, f"Episode {ep_num}")

        except Exception as e:
            messagebox.showerror("오류", f"파일 인덱싱 중 오류 발생:\n{e}")

    def on_episode_select(self, event):
        """
        [핵심 수정]
        1. 저장해둔 시작 위치로 점프 (Seek)
        2. "Episode N End"가 나올 때까지 한 줄씩 읽기 (Readline Loop)
        이 방식은 정확한 범위를 읽어오며, 인코딩 문제로부터 안전합니다.
        """
        selection = self.episode_listbox.curselection()
        if not selection:
            return

        text = self.episode_listbox.get(selection[0])
        ep_num = int(text.split()[-1])

        start_pos = self.episode_start_pos.get(ep_num)
        if start_pos is None:
            return

        content_lines = []
        end_marker = f"Episode {ep_num} End"  # 멈춰야 할 문구

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                f.seek(start_pos)  # 1. 시작점으로 점프

                while True:
                    line = f.readline()
                    if not line:
                        break

                    content_lines.append(line)

                    # 2. 종료 문구가 포함되어 있으면 멈춤
                    if end_marker in line:
                        break

            # 화면 출력
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "".join(content_lines))
            self.log_text.see(1.0)
            self.log_text.config(state="disabled")

        except Exception as e:
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"데이터 읽기 오류: {e}")
            self.log_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = MafiaLogViewerApp(root)
    root.mainloop()
