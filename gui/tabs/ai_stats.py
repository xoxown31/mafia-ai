import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import os
import numpy as np

# 스타일 설정
COLORS = {"primary": "#4285F4", "bg": "#FAFAFA", "text": "#333333", "grid": "#E0E0E0"}


class AIStatsTab:
    def __init__(self, parent, log_path):
        self.frame = ttk.Frame(parent)
        self.log_path = log_path
        self._init_ui()

    def _init_ui(self):
        top_frame = ttk.Frame(self.frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        btn_refresh = ttk.Button(top_frame, text="새로고침", command=self.refresh)
        btn_refresh.pack(side=tk.RIGHT, padx=10)

        self.chart_frame = ttk.Frame(self.frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def refresh(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        interval = 100
        episodes, win_rates = self._get_interval_win_rates(interval)

        if not episodes and len(episodes) == 0:
            ttk.Label(
                self.chart_frame, text="데이터가 부족합니다 (최소 100판 필요)"
            ).pack()
            return

        self._draw_area_chart(episodes, win_rates, interval)

    def _get_interval_win_rates(self, interval):
        if not os.path.exists(self.log_path):
            return [], []

        current_wins = 0
        count = 0
        episodes = []
        win_rates = []
        episode_num = 0

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Episode" in line and "End" in line:
                        episode_num += 1
                        is_win = bool(re.search(r"Win:\s*True", line, re.IGNORECASE))
                        if is_win:
                            current_wins += 1
                        count += 1

                        if count >= interval:
                            rate = (current_wins / count) * 100
                            episodes.append(episode_num)
                            win_rates.append(rate)
                            current_wins = 0
                            count = 0
        except Exception as e:
            print(f"Error parsing log: {e}")

        return episodes, win_rates

    def _draw_area_chart(self, x, y, interval):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])

        # 0,0 시작
        # x_line = [0] + x
        # y_line = [0.0] + y
        x_line = x
        y_line = y

        # 선 그리기
        ax.plot(x_line, y_line, color=COLORS["primary"], linewidth=2.5, marker=None)

        # 영역 채우기
        ax.fill_between(x_line, y_line, color=COLORS["primary"], alpha=0.2)
        ax.fill_between(x_line, y_line, 0, color=COLORS["primary"], alpha=0.05)

        # 실제 데이터 위치에만 마커 찍기 (0번 제외)
        scatter = ax.scatter(
            x,
            y,
            color=COLORS["primary"],
            s=50,
            zorder=3,
            edgecolors="white",
            linewidths=2,
        )

        # -----------------------------------------------------------
        # [수정됨] 축 시작점을 0으로 강제 고정하여 빈틈 없애기
        # -----------------------------------------------------------
        # ax.set_xlim(left=0)  # X축 왼쪽 끝을 0으로 고정
        ax.set_ylim(bottom=0, top=100)  # Y축 아래 끝을 0으로 고정
        # ax.margins(x=0)  # X축 좌우 여백 제거 (딱 붙게 만듦)

        # 디자인 설정
        ax.set_title(
            f"AI Win Rate Trend (Interval: {interval})",
            fontsize=14,
            fontweight="bold",
            pad=15,
            color=COLORS["text"],
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5, color=COLORS["grid"])
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 왼쪽 축(Y축)은 보이게 놔둬서 '교점' 느낌을 살림
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["grid"])

        ax.set_xlabel("Episode", fontsize=10, color=COLORS["text"])
        ax.set_ylabel("Win Rate (%)", fontsize=10, color=COLORS["text"])

        # X축 눈금 설정 (0 포함)
        if len(x_line) > 0:
            ax.set_xticks(x_line)

        # -----------------------------------------------------------
        # 마우스 오버 툴팁 (기존과 동일)
        # -----------------------------------------------------------
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        def update_annot(ind):
            idx = ind["ind"][0]
            pos_x = x[idx]
            pos_y = y[idx]

            annot.xy = (pos_x, pos_y)
            text = f"Ep: {pos_x}\nWin Rate: {pos_y:.1f}%"

            annot.set_text(text)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # 그래프의 가로/세로 중앙값 계산
            x_mid = (xlim[0] + xlim[1]) / 2
            y_mid = (ylim[0] + ylim[1]) / 2

            # 기본 오프셋 (오른쪽 위)
            offset_x = 15
            offset_y = 15
            ha = "left"  # 텍스트 가로 정렬
            va = "bottom"  # 텍스트 세로 정렬

            # 1. 점이 그래프의 '오른쪽' 절반에 있으면 -> 툴팁을 '왼쪽'으로 보냄
            if pos_x > x_mid:
                offset_x = -15
                ha = "right"

            # 2. 점이 그래프의 '위쪽' 절반에 있으면 -> 툴팁을 '아래쪽'으로 보냄
            if pos_y > y_mid:
                offset_y = -15
                va = "top"

            # 계산된 오프셋과 정렬 적용
            annot.set_position((offset_x, offset_y))
            annot.set_horizontalalignment(ha)
            annot.set_verticalalignment(va)

            # 스타일 유지
            annot.get_bbox_patch().set_facecolor(COLORS["bg"])
            annot.get_bbox_patch().set_alpha(0.9)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
