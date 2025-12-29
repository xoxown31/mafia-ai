import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MafiaLLMAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-pro"

    def get_action(self, game_status, conversation_log):
        """
        게임 상태를 받아서 AI의 발언과 행동을 결정합니다.
        """
        # 1. 프롬프트 구성 (마피아 게임의 맥락 주입)
        prompt = f"""
        당신은 지금 마피아 게임을 하고 있습니다.
        나의 플레이어 ID: {game_status['id']}
        나의 역할: {game_status['roles'][game_status['id']]}
        현재 날짜: {game_status['day']}일차
        생존 현황: {game_status['alive_status']}
        
        최근 대화 기록:
        {conversation_log}
        
        위 상황을 분석하여 다음을 결정하세요:
        1. 누구를 마피아로 의심하거나, 자신을 어떤 역할로 주장할 것인가?
        2. 누구에게 투표할 것인가?
        
        형식: JSON ({{ "claim_role": 0, "target_id": 1, "speech": "..." }})
        """

        # 2. API 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 냉철한 마피아 게임 전략가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},  # JSON 출력을 강제함
        )

        return response.choices[0].message.content
