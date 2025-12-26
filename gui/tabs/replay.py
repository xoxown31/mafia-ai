import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import re


class ReplayTab:
    def __init__(self, parent, log_path):
        self.frame = ttk.Frame(parent)
        self.log_path = log_path
        self.episode_start_pos = {}

        # [신규] 역할 정보와 툴팁을 관리할 변수 추가
        self.role_map = {}  # {player_id: role_name}
        self.tooltip_win = None

        self._init_ui()
        self.refresh_list()

    def _init_ui(self):
        paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned, width=250)
        paned.add(left, weight=1)

        header = ttk.Frame(left)
        header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(header, text="에피소드 목록").pack(side=tk.LEFT)
        ttk.Button(header, text="새로고침", width=8, command=self.refresh_list).pack(
            side=tk.RIGHT
        )

        scroll = ttk.Scrollbar(left)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            left, yscrollcommand=scroll.set, font=("Consolas", 11)
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(paned)
        paned.add(right, weight=4)

        ttk.Label(right, text="게임 로그").pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(
            right, font=("맑은 고딕", 10), state="disabled", cursor="arrow"
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

    def refresh_list(self):
        self.episode_start_pos = {}
        self.listbox.delete(0, tk.END)
        if not os.path.exists(self.log_path):
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    match = re.search(r"Episode (\d+) Start", line)
                    if match:
                        ep = int(match.group(1))
                        self.episode_start_pos[ep] = pos
                        self.listbox.insert(tk.END, f"Episode {ep}")
        except Exception as e:
            messagebox.showerror("오류", f"인덱싱 오류: {e}")

    def on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        ep = int(self.listbox.get(sel[0]).split()[-1])
        start = self.episode_start_pos.get(ep)

        self.role_map = {}

        lines = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                f.seek(start)
                while True:
                    line = f.readline()
                    if not line:
                        break

                    # [신규] 로그 초반부에서 역할 정보 파싱
                    # 예: "플레이어 1: MAFIA (Rational Agent)"
                    role_match = re.search(r"플레이어 (\d+): (\w+)", line)
                    if role_match:
                        pid = int(role_match.group(1))
                        role = role_match.group(2)
                        self.role_map[pid] = role

                    lines.append(line)
                    if f"Episode {ep} End" in line:
                        break

            self.text_area.config(state="normal")
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "".join(lines))

            # [신규] 텍스트 내 플레이어 이름에 태그 달기 함수 호출
            self._tag_players_in_log()

            self.text_area.see(1.0)
            self.text_area.config(state="disabled")
        except Exception as e:
            print(f"Error reading log: {e}")

    def _tag_players_in_log(self):
        """로그 텍스트에서 '플레이어 N' 또는 'AI(N)'을 찾아 태그를 달고 툴팁 이벤트를 연결합니다."""

        # 0번부터 7번 플레이어까지 각각 처리
        for pid in range(8):
            tag_name = f"player_{pid}"

            # 검색할 패턴들 (한글, AI 표기, 영어 표기 대응)
            patterns = [
                f"플레이어 {pid}",  # "플레이어 6"
                f"AI\\({pid}\\)",  # "AI(0)"
                f"Player {pid}",  # "Player 6" (혹시 모를 경우 대비)
            ]

            for pattern in patterns:
                start_index = "1.0"
                while True:
                    # 정규식 검색 (길이를 알기 위해 count 변수 사용)
                    count_var = tk.IntVar()
                    pos = self.text_area.search(
                        pattern,
                        start_index,
                        stopindex=tk.END,
                        count=count_var,
                        regexp=True,
                    )

                    if not pos:
                        break

                    # 끝 위치 계산 ("행.열 + 글자수c")
                    end_index = f"{pos}+{count_var.get()}c"

                    # 태그 추가
                    self.text_area.tag_add(tag_name, pos, end_index)

                    # 다음 검색 시작 위치 갱신
                    start_index = end_index

            # 태그 스타일 설정 (파란색 + 밑줄)
            self.text_area.tag_config(tag_name, foreground="blue", underline=True)

            # 마우스 이벤트 바인딩 (Enter: 진입, Leave: 나감)
            # 주의: lambda에서 pid 값을 고정(capture)하기 위해 p=pid 사용
            self.text_area.tag_bind(
                tag_name, "<Enter>", lambda e, p=pid: self.show_tooltip(e, p)
            )
            self.text_area.tag_bind(tag_name, "<Leave>", lambda e: self.hide_tooltip())

    def show_tooltip(self, event, pid):
        """마우스 위치에 직업 정보 툴팁 표시"""
        self.hide_tooltip()  # 기존 툴팁 제거

        role = self.role_map.get(pid, "Unknown")

        # 직업별 배경색 지정 (시각적 구분)
        bg_color = "#FFFFE0"  # 기본 (연노랑)
        if role == "MAFIA":
            bg_color = "#FFCCCC"  # 마피아 (연빨강)
        elif role == "POLICE":
            bg_color = "#CCEAFF"  # 경찰 (연파랑)
        elif role == "DOCTOR":
            bg_color = "#CCFFCC"  # 의사 (연초록)

        # 툴팁 윈도우 생성 (타이틀바 없는 팝업창)
        self.tooltip_win = tk.Toplevel(self.text_area)
        self.tooltip_win.wm_overrideredirect(True)

        # 마우스 커서보다 약간 아래/오른쪽에 위치
        self.tooltip_win.geometry(f"+{event.x_root + 15}+{event.y_root + 10}")

        label = tk.Label(
            self.tooltip_win,
            text=f"Player {pid}: {role}",
            justify="left",
            background=bg_color,
            relief="solid",
            borderwidth=1,
            font=("맑은 고딕", 9),
        )
        label.pack()

    def hide_tooltip(self):
        """툴팁 제거"""
        if self.tooltip_win:
            self.tooltip_win.destroy()
            self.tooltip_win = None
