import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

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

    # --- 3. 시각화 (Visualization) ---
    plt.figure(figsize=(15, 6))

    # (1) 진영별 승률 (Pie Chart)
    plt.subplot(1, 2, 1)
    labels = team_wins.keys()
    sizes = team_wins.values()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title('Win Rate by Team (Mafia vs Citizen)')

    # (2) 성격별 승률 (Bar Chart)
    plt.subplot(1, 2, 2)
    chars = []
    win_rates = []
    
    for char, stat in char_stats.items():
        if stat["total"] > 0:
            chars.append(char)
            win_rates.append(stat["wins"] / stat["total"] * 100)
    
    # 막대 그래프 그리기
    bars = plt.bar(chars, win_rates, color=['#a3c1ad', '#f4a261', '#e76f51', '#264653'])
    plt.ylim(0, 100)
    plt.xlabel('Character Type')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate by Character Personality')
    plt.grid(axis='y', alpha=0.3)
    
    # 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"[Analysis] 분석 결과 그래프 저장 완료: {output_img}")