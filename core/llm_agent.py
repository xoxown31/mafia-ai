import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-pro"

    def get_action(self, game_status, conversation_log):
        # 1. 게임 상태와 로그를 LLM 프롬프트에 맞게 가공
        my_id = game_status.get("id", -1)
        roles = game_status.get("roles", [])
        my_role_id = roles[my_id] if 0 <= my_id < len(roles) else 0
        input_data = {
            "my_id": my_id,
            "my_role_name": self._get_role_name(my_role_id),
            "current_day": game_status.get("day", 1),
            "alive_players": game_status.get("alive_status", []),
            "last_execution": game_status.get(
                "last_execution_result"
            ),  # 어제 처형된 사람 정보
            "last_night_kill": game_status.get(
                "last_night_kill"
            ),  # 어젯밤 죽은 사람 정보
            "mafia_team": game_status.get(
                "mafia_team_members"
            ),  # (마피아일 경우) 동료 목록
            "police_results": game_status.get(
                "police_investigation_results"
            ),  # (경찰일 경우) 조사 결과
            "discussion_history": conversation_log,
        }

        # 2. 프롬프트 정의
        prompt = f"""
        ### SYSTEM ROLE
        당신은 냉철한 마피아 게임 전략가이자 분석가입니다. 
        제공된 [GAME STATE]를 바탕으로 다른 플레이어에 대한 신념을 업데이트하고, 최적의 전략적 행동을 결정하십시오.
        각 플레이어의 번호(ID)를 반드시 고려하십시오.

        ### [GAME STATE]
        {input_data}

        ### TASK 1: Belief Update
        각 플레이어의 정체성에 대한 변화량($\Delta$)을 -100에서 100 사이의 정수로 계산하십시오.
        - 논리적 모순을 보이면 마피아 확률($\Delta$) 증가.
        - 본인({input_data['my_id']})은 제외.

        ### TASK 2: Action Strategy
        최종 결정 수식 $L_i = S_{{belief, i}} + Bias_i$를 바탕으로 전략적 편향($Bias$)을 설정하십시오.

        - [시민 팀 (시민, 경찰, 의사)]일 경우: 
        실제로 마피아라고 의심되는 플레이어에게 양수의 $Bias$를 부여하여 처형 확률을 높이십시오.
        경찰이라면 조사 결과 마피아로 판명된 자에게 가장 높은 $Bias$를 부여하십시오.
        
        - [마피아 팀]일 경우: 
        자신의 정체를 숨기기 위해 무고한 시민을 몰아가거나(선동), 아군 마피아를 보호하는 방향으로 $Bias$를 조작하십시오(기만).
        
        - 공통: 발언이 위험하거나 상황을 관망해야 할 때는 $Silence\_Bias$를 높이십시오.

        ### OUTPUT FORMAT (JSON)
        반드시 아래의 JSON 구조만 출력하십시오:
        {{
            "belief_updates": [
                {{ "player_id": int, "role": "MAFIA", "delta": int, "reason": "string" }}
            ],
            "action_strategy": {{
                "biases": [float, float, float, float, float, float, float, float], // 플레이어 0~7에 대한 가중치
                "silence_bias": float,  // 아무도 지목하지 않는 행동에 대한 가중치
                "strategy_note": "string" //이 가중치를 설정한 전략적 의도 (예: 마피아로서 경찰인 3번을 모함하기 위해 가중치 부여)
            }},
            "speech": "토론 시간에 실제로 내뱉을 한국어 대사"
        }}
        """

        # 3. API 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 이모지를 사용하지 않는 냉철한 게임 전략가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def get_night_action(self, game_status, conversation_log):
        my_id = game_status.get("id", -1)
        roles = game_status.get("roles", [])
        my_role_id = roles[my_id] if 0 <= my_id < len(roles) else 0
        role_name = self._get_role_name(my_role_id)

        input_data = {
            "my_id": my_id,
            "my_role_name": role_name,
            "current_day": game_status.get("day", 1),
            "alive_players": game_status.get("alive_status", []),
            "mafia_team": game_status.get("mafia_team_members"),
            "police_results": game_status.get("police_investigation_results"),
            "discussion_history": conversation_log,
        }

        # 밤 단계 전용 프롬프트
        prompt = f"""
        ### SYSTEM ROLE
        당신은 마피아 게임의 밤 단계 전략가입니다. 당신의 역할은 [{role_name}]입니다.
        낮의 토론 내용과 현재까지의 정보를 바탕으로 팀의 승리를 위한 최적의 타겟을 선정하십시오.

        ### [GAME STATE]
        {input_data}

        ### ROLE-SPECIFIC STRATEGY
        - [마피아]: 시민 팀의 핵심 인물(경찰/의사로 의심되는 자)을 제거하십시오. 동료 마피아를 지목하지 않도록 주의하십시오.
        - [경찰]: 마피아로 가장 의심되는 생존자를 조사하십시오. 이미 결과를 아는 플레이어는 피하십시오.
        - [의사]: 마피아가 공격할 가능성이 가장 높은 중요한 인물(본인 포함)을 보호하십시오.

        ### OUTPUT FORMAT (JSON)
        {{
            "target_id": int,
            "reasoning": "이 타겟을 선정한 전략적 이유",
            "confidence": int (0-100 사이의 확신도)
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 조용하고 치밀하게 밤의 행동을 결정하는 전략가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _get_role_name(self, role_int):
        # config의 역할을 문자열로 매핑하는 헬퍼 함수
        # 주의: 이 매핑은 실제 config.py의 역할 설정과 일치해야 합니다.
        mapping = {0: "시민", 1: "마피아", 2: "경찰", 3: "의사"}
        return mapping.get(role_int, "알 수 없음")
