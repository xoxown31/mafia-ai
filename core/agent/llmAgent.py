import os
import json
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

from core.agent.baseAgent import BaseAgent
from config import config, Role, Phase, EventType
from state import GameStatus, GameEvent

load_dotenv()


class LLMAgent(BaseAgent):
    def __init__(self, player_id: int, role: Role = Role.CITIZEN):
        super().__init__(player_id, role)

        # API 설정
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-mini"

    def update_belief(self, history: List[GameEvent]):
        """객관적 사실(Fact) 반영"""
        for event in history:
            # 경찰 조사 결과 (시스템 공개 포함)
            if event.event_type == EventType.POLICE_RESULT:
                target_id = event.target_id
                role_found = event.value # Role Enum
                
                if target_id is not None and isinstance(role_found, Role):
                    role_idx = int(role_found)
                    self.belief[target_id, role_idx] = 100.0
                    # 다른 역할일 확률은 -100으로
                    for r in Role:
                        if r != role_idx:
                            self.belief[target_id, r] = -100.0

            # 처형 결과 (성공/실패 여부만 확인, 역할 공개는 POLICE_RESULT로 처리됨)
            elif event.event_type == EventType.EXECUTE:
                pass

    def get_action(self, conversation_log: str) -> str:
        """주관적 추론(Hunch) 및 결정"""
        if not self.current_status:
            return json.dumps({"error": "No game status observed"})

        phase = self.current_status.phase

        if phase == Phase.DAY_DISCUSSION:
            return self._handle_discussion(conversation_log)
        elif phase == Phase.DAY_VOTE:
            return self._handle_vote(conversation_log)
        elif phase == Phase.DAY_EXECUTE:
            return self._handle_final_vote(conversation_log)
        elif phase == Phase.NIGHT:
            return self._handle_night(conversation_log)

        return json.dumps({"error": f"Unknown phase: {phase}"})

    # --- 내부 핸들러 메서드 ---

    def _handle_discussion(self, conversation_log: str) -> str:
        """낮 토론: 신념 분석 -> 전략 수립 -> 대사 생성"""
        # Pydantic 모델을 dict로 변환하여 프롬프트에 주입
        status_json = self.current_status.model_dump_json(exclude_none=True)
        
        prompt = f"""
### 당신의 페르소나
- 당신의 ID: {self.id}
- 당신의 실제 역할: [{self.role.name}]
- **절대 원칙**: 당신은 자신의 팀(시민 혹은 마피아)의 승리를 위해 행동해야 합니다. 스스로를 마피아라고 주장하거나 동료를 투표하는 것은 금지됩니다.

### 논리적 사고 단계
1. **분석**: 현재 생존자 중 누가 가장 의심스러운가?
2. **검증**: 내가 공격하려는 대상이 내 팀원인가? (마피아팀인 경우 mafia_members 확인)
3. **결정**: 어떤 주장을 할 것인가?

### [GAME STATE]
{status_json}

### [CONVERSATION LOG]
{conversation_log}

### [BELIEF MATRIX SUMMARY]
{self.belief.tolist()}

### 출력 포맷 (반드시 JSON 형식 준수)
{{
    "belief_updates": [{{"player_id": int, "role": "MAFIA"|"POLICE"|"DOCTOR"|"CITIZEN", "delta": int}}],
    "action_strategy": {{
        "biases": [float], "silence_bias": float, "strategy_note": "전략 요약"
    }},
    "claim": int (0: 내 직업 주장, 1: 남의 직업 주장, 2: 침묵/기타 발언),
    "target_id": int or null,
    "role": int or null (주장하는 직업 번호: 0:시민, 1:경찰, 2:의사, 3:마피아),
    "reason": "왜 그렇게 주장했는지에 대한 논리적 근거"
    "discussion_status": "Proceed"|"End"
}}
"""
        response_json = self._call_llm(prompt)
        self._apply_llm_belief_updates(response_json)
        return response_json

    def _handle_vote(self, conversation_log: str) -> str:
        """낮 투표: 최적의 처형 대상 선정"""
        status_json = self.current_status.model_dump_json(exclude_none=True)

        prompt = f"""
### TASK: DAY VOTE
당신은 [{self.role.name}]입니다. 오늘 누구를 처형할지 결정하십시오.
[상태]: {status_json}
[대화]: {conversation_log}
-1은 기권을 의미합니다.

### [OUTPUT FORMAT (JSON)]
{{
    "target_id": int, (0 to 7, -1 for abstain)
    "reason": "투표 이유"
}}
"""
        return self._call_llm(prompt)

    def _handle_final_vote(self, conversation_log: str) -> str:
        """사형 찬반 투표"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        # execution_target 정보는 conversation_log나 status 어딘가에 있어야 함.
        # 현재 GameStatus 구조상 action_history 등을 통해 유추하거나 별도 필드가 필요할 수 있음.
        # 여기서는 status에 포함되어 있다고 가정하거나, conversation_log를 통해 판단.
        
        prompt = f"""
### TASK: FINAL VOTE (찬반 투표)
당신은 [{self.role.name}]입니다.
처형 후보에 대한 찬반을 최종 결정하십시오.

### [FULL GAME STATE]
{status_json}

### INSTRUCTIONS
- 제공된 게임 상태와 당신의 신념 매트릭스를 종합하여 합리적인 결정을 내리십시오.
- 당신이 이전에 그를 마피아라고 의심하여 투표했다면, 일관성 있게 찬성표를 던지는 것이 논리적입니다.
- 만약 그가 당신의 팀원(예: 같은 마피아)이거나, 무고한 시민이라는 확신이 있다면 반대하십시오.

### [OUTPUT FORMAT (JSON)]
{{
    "agree_execution": int (1: 찬성, -1: 반대, 0: 기권)
}}
"""
        return self._call_llm(prompt)

    def _handle_night(self, conversation_log: str) -> str:
        """밤 행동: 특수 능력 수행"""
        status_json = self.current_status.model_dump_json(exclude_none=True)

        prompt = f"""
### TASK: NIGHT ACTION (밤 행동)
- **당신의 역할**: [{self.role.name}]
- **당신의 ID**: {self.id}

### [GAME STATE]
{status_json}

### INSTRUCTIONS
1.  **당신의 역할에 맞는 행동을 수행하십시오.**
    -   **마피아**: `players` 목록 중 살아있는 플레이어(is_alive=True) 중에서만 제거할 대상을 선택하십시오. 당신의 팀원도 선택하지 마십시오.
    -   **경찰**: 살아있는 플레이어 중에서만 조사할 대상을 선택하십시오.
    -   **의사**: 살아있는 플레이어 중에서만 보호할 대상을 선택하십시오. (자신 포함 가능)
2.  **선택한 대상의 ID를 `target_id`에 명시하십시오.**

### [OUTPUT FORMAT (JSON)]
{{
    "target_id": int
}}
"""
        response_json = self._call_llm(prompt)
        # 밤 행동에 대한 추론 결과도 신념에 반영할 수 있다면 여기서 처리
        return response_json

    def _call_llm(self, prompt: str) -> str:
        """Upstage API 호출 및 예외 처리"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 마피아 게임을 하는 AI 플레이어입니다. 당신의 역할과 생각은 프롬프트에 주어집니다. 반드시 당신에게 주어진 역할에 입각해서만 추론하고 JSON 형식으로만 답변해야 합니다. 다른 사람의 역할을 흉내내지 마십시오.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            # print(response.choices[0].message.content, end="\n\n")
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _apply_llm_belief_updates(self, raw_json: str):
        """LLM 분석 결과를 신념 행렬에 반영"""
        try:
            data = json.loads(raw_json)
            updates = data.get("belief_updates", [])
            for up in updates:
                p_id = up.get("player_id")
                role_str = up.get("role")
                delta = up.get("delta", 0)
                
                if p_id is not None and role_str in Role.__members__:
                    role_enum = Role[role_str]
                    col_idx = int(role_enum)
                    if 0 <= p_id < config.game.PLAYER_COUNT:
                        self.belief[p_id, col_idx] = np.clip(
                            self.belief[p_id, col_idx] + delta, -100, 100
                        )
        except Exception:
            pass
