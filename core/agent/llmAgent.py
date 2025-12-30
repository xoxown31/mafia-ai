import os
import json
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 구조에 따라 import 경로는 수정이 필요할 수 있습니다.
from core.agent.baseAgent import BaseAgent
import config

load_dotenv()


class LLMAgent(BaseAgent):
    def __init__(self, player_id: int, role: int = config.ROLE_CITIZEN):
        # 부모 클래스(BaseAgent) 초기화: self.id, self.role, self.belief(N x 4 행렬) 생성
        super().__init__(player_id, role)

        # API 설정
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-mini"

        # 내부 역할 매핑 표준화 (신념 행렬 인덱스와 일치시킴)
        # 0: CITIZEN, 1: POLICE, 2: DOCTOR, 3: MAFIA
        self.role_map = {"CITIZEN": 0, "POLICE": 1, "DOCTOR": 2, "MAFIA": 3}
        self.inv_role_map = {v: k for k, v in self.role_map.items()}

    def update_belief(self, game_status: Dict):
        """
        BaseAgent의 추상 메서드 구현.
        게임 엔진에서 확정된 정보(사망자 등)가 있을 때 수치적으로 기본 업데이트를 수행합니다.
        """
        pass

    def get_action(self, game_status: Dict, conversation_log: str) -> str:
        """
        [핵심 함수] Phase별로 분기하여 적절한 프롬프트를 실행하고 결과를 반환합니다.
        """
        phase = game_status.get("phase")

        if phase == config.PHASE_DAY_DISCUSSION:
            return self._handle_discussion(game_status, conversation_log)
        elif phase == config.PHASE_DAY_VOTE:
            return self._handle_vote(game_status, conversation_log)
        elif phase == config.PHASE_DAY_EXECUTE:
            return self._handle_final_vote(game_status, conversation_log)
        elif phase == config.PHASE_NIGHT:
            return self._handle_night(game_status, conversation_log)

        return json.dumps({"error": f"Unknown phase: {phase}"})

    # --- 내부 핸들러 메서드 ---

    def _handle_discussion(self, game_status: Dict, conversation_log: str) -> str:
        """낮 토론: 신념 분석 -> 전략 수립 -> 대사 생성"""
        input_data = self._prepare_input_data(game_status, conversation_log)

        prompt = f"""
### ROLE: 마피아 게임 전략가 (단계: 낮 토론)
당신은 Player {self.id}이며, 실제 직업은 [{input_data['my_role_name']}]입니다.

### [GAME STATE]
{input_data}

### INSTRUCTIONS
1. **Belief Update**: 타인의 마피아 확률 변화량($\Delta$)을 -100~100 사이로 계산.
2. **Strategy**: $L_i = S_{{belief, i}} + Bias_i$ 기반으로 타겟 편향($Bias$) 설정.
3. **Claiming Logic (절대 준수)**: 
   - **내가 누구인지 밝힐 때**: "claim": 0, "target_id": {self.id}, "role": (주장할 직업 번호)
   - **남을 지목할 때 (예: "3번이 마피아다")**: "claim": 1, "target_id": (대상 ID), "role": (그 사람의 예상 직업 번호)
   - **일반적인 대화/침묵**: "claim": 2, "target_id": null, "role": null
4. **직업 번호**: 0:시민, 1:경찰, 2:의사, 3:마피아

### OUTPUT FORMAT (Strict JSON)
{{
    "belief_updates": [{{"player_id": int, "role": "CITIZEN"|"POLICE"|"DOCTOR"|"MAFIA", "delta": int}}],
    "action_strategy": {{
        "biases": [8 floats],
        "silence_bias": float(0-100),
        "strategy_note": "STR"
    }},
    "discussion_status": "Proceed/End",
    "silence": bool,
    "claim": 0|1|2,
    "target_id": int|null,
    "role": 0|1|2|3|null
}}
"""
        response_json = self._call_llm(prompt)
        self._apply_belief_updates(response_json)
        return response_json

    def _handle_vote(self, game_status: Dict, conversation_log: str) -> str:
        """낮 투표: 최적의 처형 대상 선정"""
        input_data = self._prepare_input_data(game_status, conversation_log)

        prompt = f"""
### TASK: DAY VOTE
당신은 [{input_data['my_role_name']}]입니다. 오늘 누구를 처형할지 결정하십시오.
[상태]: {input_data}
-1은 기권을 의미합니다.

### [OUTPUT FORMAT (JSON)]
{{
    "target_id": int, (0 to 7, -1 for abstain)
    "reason": "투표 이유"
}}
"""
        return self._call_llm(prompt)

    def _handle_final_vote(self, game_status: Dict, conversation_log: str) -> str:
        """사형 찬반 투표"""
        input_data = self._prepare_input_data(game_status, conversation_log)
        target = game_status.get("execution_target")

        prompt = f"""
### TASK: FINAL VOTE
플레이어 {target}의 처형 찬반을 결정하십시오.
당신의 역할: {input_data['my_role_name']}

### [OUTPUT FORMAT (JSON)]
{{
    "agree_execution": int (1: 찬성, -1: 반대, 0: 기권)
}}
"""
        return self._call_llm(prompt)

    def _handle_night(self, game_status: Dict, conversation_log: str) -> str:
        """밤 행동: 특수 능력 수행"""
        input_data = self._prepare_input_data(game_status, conversation_log)

        prompt = f"""
### TASK: NIGHT ACTION
당신의 역할 [{input_data['my_role_name']}]에 따라 행동하십시오.
- 마피아: 시민 제거, 경찰: 마피아 조사, 의사: 보호

### [OUTPUT FORMAT (JSON)]
{{
    "belief_updates": [
        {{"player_id": int, "role": "MAFIA"|"CITIZEN"|"POLICE"|"DOCTOR", "delta": int}}
    ],
    "target_id": int(0-7)
}}
"""
        response_json = self._call_llm(prompt)
        self._apply_belief_updates(response_json)
        return response_json

    # --- 유틸리티 및 헬퍼 메서드 ---

    def _prepare_input_data(self, game_status: Dict, conversation_log: str) -> Dict:
        """프롬프트 주입용 데이터 가공"""
        return {
            "my_id": self.id,
            "my_role_name": self._get_role_name(self.role),
            "current_day": game_status.get("day", 1),
            "alive_players": game_status.get("alive_status", []),
            "last_execution_result": game_status.get("last_execution_result"),
            "last_night_result": game_status.get("last_night_result"),
            "conversation_history": conversation_log,
            "belief_matrix_summary": self.belief.tolist(),
        }

    def _call_llm(self, prompt: str) -> str:
        """Upstage API 호출 및 예외 처리"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 냉철한 마피아 게임 전략가입니다. 반드시 JSON으로만 답변하십시오.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            print(response.choices[0].message.content, end="\n\n")
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _apply_belief_updates(self, raw_json: str):
        """LLM 분석 결과를 신념 행렬에 반영"""
        try:
            data = json.loads(raw_json)
            updates = data.get("belief_updates", [])
            for up in updates:
                p_id = up.get("player_id")
                role_key = up.get("role")
                delta = up.get("delta", 0)
                if (
                    p_id is not None
                    and 0 <= p_id < config.PLAYER_COUNT
                    and role_key in self.role_map
                ):
                    col_idx = self.role_map[role_key]
                    self.belief[p_id, col_idx] = np.clip(
                        self.belief[p_id, col_idx] + delta, -100, 100
                    )
        except Exception:
            pass

    def _get_role_name(self, role_int: int) -> str:
        mapping = {0: "시민", 1: "경찰", 2: "의사", 3: "마피아"}
        return mapping.get(role_int, "알 수 없음")
