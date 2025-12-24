import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns

def analyze_log_file(log_path, output_img="analysis_detailed.png"):
    print(f"[Analysis] 로그 파일 분석 중: {log_path}")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    games = []
    current_game = {'roles': {}, 'chars': {}, 'winner': None, 'duration': 0}
    
    # --- 1. 로그 파싱 (Parsing) ---
    for line in lines:
        line = line.strip()
        
        # 에피소드 시작 감지
        if "Episode" in line and "Start" in line:
            current_game = {'roles': {}, 'chars': {}, 'winner': None, 'duration': 0}

        # 플레이어 정보 추출 (예: 플레이어 1: CITIZEN (CopyCat))
        # Regex: 플레이어 (숫자): (직업) ((성격))
        match = re.search(r"플레이어 (\d+): (\w+) \((\w+)\)", line)
        if match:
            pid = int(match.group(1))
            role = match.group(2)
            char = match.group(3)
            current_game['roles'][pid] = role
            current_game['chars'][pid] = char
            
        # 게임 종료 및 승자 판별
        if "게임 종료" in line:
            if "시민 팀 승리" in line:
                current_game['winner'] = "Citizen"
            elif "마피아 승리" in line:
                current_game['winner'] = "Mafia"
            
            # 게임 길이 (마지막 Day 정보가 기록되었을 것이므로 로그의 Day {N} 확인 필요)
            # 여기서는 편의상 게임 객체를 리스트에 추가
            games.append(current_game)

    if not games:
        print("[Analysis] 분석할 게임 데이터가 없습니다.")
        return

    print(f"[Analysis] 총 {len(games)}개의 에피소드 데이터 추출 완료.")

    # --- 2. 통계 집계 (Aggregation) ---
    team_wins = {"Mafia": 0, "Citizen": 0}
    char_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    # 성격별 x 직업별 승률 통계
    char_role_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
    
    for game in games:
        winner_team = game['winner']
        if not winner_team:
            continue
            
        team_wins[winner_team] += 1
        
        # 성격별 승률 계산
        # (해당 게임에 참여한 모든 플레이어에 대해, 자신이 속한 팀이 이겼는지 확인)
        for pid, role in game['roles'].items():
            char = game['chars'].get(pid, "Unknown")
            
            # 플레이어의 팀 확인
            my_team = "Mafia" if role == "MAFIA" else "Citizen"
            
            char_stats[char]["total"] += 1
            if my_team == winner_team:
                char_stats[char]["wins"] += 1
            
            # 성격별 x 직업별 통계
            char_role_stats[char][role]["total"] += 1
            if my_team == winner_team:
                char_role_stats[char][role]["wins"] += 1

    # --- 3. 시각화 (Visualization) ---
    plt.figure(figsize=(15, 6))

    # (1) 진영별 승률 (Pie Chart)
    plt.subplot(1, 2, 1)
    labels = team_wins.keys()
    sizes = team_wins.values()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title('Win Rate by Team (Mafia vs Citizen)')

    # (2) 성격별 x 직업별 승률 (Heatmap)
    plt.subplot(1, 2, 2)
    
    # 모든 성격과 직업 수집
    all_chars = sorted(list(char_role_stats.keys()))
    all_roles = sorted(list(set(role for char_data in char_role_stats.values() for role in char_data.keys())))
    
    # 히트맵 데이터 생성 (성격 x 직업)
    heatmap_data = np.zeros((len(all_chars), len(all_roles)))
    
    for i, char in enumerate(all_chars):
        for j, role in enumerate(all_roles):
            stat = char_role_stats[char][role]
            if stat["total"] > 0:
                heatmap_data[i, j] = stat["wins"] / stat["total"] * 100
            else:
                heatmap_data[i, j] = np.nan  # 데이터 없는 경우
    
    # 히트맵 그리기
    sns.heatmap(heatmap_data, 
                xticklabels=all_roles, 
                yticklabels=all_chars,
                annot=True,  # 셀에 수치 표시
                fmt='.1f',   # 소수점 한 자리까지 표시
                cmap='YlOrRd',  # 색상 맵
                cbar_kws={'label': 'Win Rate (%)'},
                vmin=0, 
                vmax=100,
                linewidths=0.5,
                linecolor='gray')
    
    plt.xlabel('Role')
    plt.ylabel('Character Type')
    plt.title('Win Rate by Character × Role')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"[Analysis] 분석 결과 그래프 저장 완료: {output_img}")