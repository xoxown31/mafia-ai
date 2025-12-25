import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns


def parse_game_data(log_path):
    """
    [NEW] 로그 파일을 읽어 게임 데이터 리스트를 반환하는 함수.
    GUI 등 외부 프로그램에서 재사용하기 위해 분리함.
    """
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    games = []
    current_game = {"roles": {}, "chars": {}, "winner": None, "duration": 0}

    # --- 로그 파싱 (Parsing) ---
    for line in lines:
        line = line.strip()

        # 에피소드 시작 감지
        if "Episode" in line and "Start" in line:
            current_game = {"roles": {}, "chars": {}, "winner": None, "duration": 0}

        # 플레이어 정보 추출 (예: 플레이어 1: CITIZEN (CopyCat))
        match = re.search(r"플레이어 (\d+): (\w+) \((\w+)\)", line)
        if match:
            pid = int(match.group(1))
            role = match.group(2)
            char = match.group(3)
            current_game["roles"][pid] = role
            current_game["chars"][pid] = char

        # 게임 종료 및 승자 판별
        if "게임 종료" in line:
            if "시민 팀 승리" in line:
                current_game["winner"] = "Citizen"
            elif "마피아 승리" in line:
                current_game["winner"] = "Mafia"

            # 게임 데이터를 리스트에 추가
            games.append(current_game)

    return games


def analyze_log_file(log_path, output_img="analysis_detailed.png"):
    print(f"[Analysis] 로그 파일 분석 중: {log_path}")

    # [수정] 위에서 만든 함수를 호출하여 데이터를 가져옴
    games = parse_game_data(log_path)

    if not games:
        print("[Analysis] 분석할 게임 데이터가 없습니다.")
        return

    print(f"[Analysis] 총 {len(games)}개의 에피소드 데이터 추출 완료.")

    # --- 2. 통계 집계 (Aggregation) ---
    team_wins = {"Mafia": 0, "Citizen": 0}
    char_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    char_role_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    for game in games:
        winner_team = game["winner"]
        if not winner_team:
            continue

        team_wins[winner_team] += 1

        # 성격별 승률 계산
        for pid, role in game["roles"].items():
            char = game["chars"].get(pid, "Unknown")
            my_team = "Mafia" if role == "MAFIA" else "Citizen"

            char_stats[char]["total"] += 1
            if my_team == winner_team:
                char_stats[char]["wins"] += 1

            char_role_stats[char][role]["total"] += 1
            if my_team == winner_team:
                char_role_stats[char][role]["wins"] += 1

    # --- 3. 시각화 (Visualization) ---
    plt.figure(figsize=(15, 6))

    # (1) 진영별 승률 (Pie Chart)
    plt.subplot(1, 2, 1)
    labels = team_wins.keys()
    sizes = team_wins.values()
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#ff9999", "#66b3ff"],
    )
    plt.title("Win Rate by Team (Mafia vs Citizen)")

    # (2) 성격별 x 직업별 승률 (Heatmap)
    plt.subplot(1, 2, 2)

    all_chars = sorted(list(char_role_stats.keys()))
    all_roles = sorted(
        list(
            set(
                role
                for char_data in char_role_stats.values()
                for role in char_data.keys()
            )
        )
    )

    heatmap_data = np.zeros((len(all_chars), len(all_roles)))

    for i, char in enumerate(all_chars):
        for j, role in enumerate(all_roles):
            stat = char_role_stats[char][role]
            if stat["total"] > 0:
                heatmap_data[i, j] = stat["wins"] / stat["total"] * 100
            else:
                heatmap_data[i, j] = np.nan

    sns.heatmap(
        heatmap_data,
        xticklabels=all_roles,
        yticklabels=all_chars,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Win Rate (%)"},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="gray",
    )

    plt.xlabel("Role")
    plt.ylabel("Character Type")
    plt.title("Win Rate by Character × Role")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"[Analysis] 분석 결과 그래프 저장 완료: {output_img}")
