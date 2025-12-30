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
### 당신의 페르소나
- 당신의 ID: {self.id}
- 당신의 실제 역할: [{input_data.get('my_role_name')}]
- **절대 원칙**: 당신은 자신의 팀(시민 혹은 마피아)의 승리를 위해 행동해야 합니다. 스스로를 마피아라고 주장하거나 동료를 투표하는 것은 금지됩니다.

### 논리적 사고 단계
1. **분석**: 현재 생존자 중 누가 가장 의심스러운가?
2. **검증**: 내가 공격하려는 대상이 내 팀원인가? (마피아팀인 경우 mafia_members 확인)
3. **결정**: 어떤 주장을 할 것인가?

### [GAME STATE]
{json.dumps(input_data, indent=2, ensure_ascii=False)}

### 출력 포맷 (반드시 JSON 형식 준수)
{{
    "belief_updates": [{{"player_id": int, "role": "MAFIA", "delta": int}}],
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
### TASK: FINAL VOTE (찬반 투표)
당신은 [{input_data.get('my_role_name')}]입니다.
토론과 1차 투표 결과, 플레이어 {target}이(가) 최다 득표를 하여 처형 후보가 되었습니다.
이제 플레이어 {target}의 처형에 대한 찬반을 최종 결정하십시오.

### [FULL GAME STATE]
{json.dumps(input_data, indent=2, ensure_ascii=False)}

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

    def _handle_night(self, game_status: Dict, conversation_log: str) -> str:
        """밤 행동: 특수 능력 수행"""
        input_data = self._prepare_input_data(game_status, conversation_log)

        prompt = f"""
### TASK: NIGHT ACTION (밤 행동)
- **당신의 역할**: [{input_data.get('my_role_name')}]
- **당신의 ID**: {self.id}

### [GAME STATE]
{json.dumps(input_data, indent=2, ensure_ascii=False)}

### INSTRUCTIONS
1.  **당신의 역할에 맞는 행동을 수행하십시오.**
    -   **마피아**: `alive_players` 목록에 있는 플레이어 중에서만 제거할 대상을 선택하십시오. 죽은 사람은 선택할 수 없습니다. 당신의 팀원도 선택하지 마십시오.
    -   **경찰**: `alive_players` 목록에 있는 플레이어 중에서만 조사할 대상을 선택하십시오.
    -   **의사**: `alive_players` 목록에 있는 플레이어 중에서만 보호할 대상을 선택하십시오. (자신 포함 가능)
2.  **선택한 대상의 ID를 `target_id`에 명시하십시오.**

### [OUTPUT FORMAT (JSON)]
{{
    "target_id": int
}}
"""
        response_json = self._call_llm(prompt)
        self._apply_belief_updates(response_json)
        return response_json

    # --- 유틸리티 및 헬퍼 메서드 ---

    def _prepare_input_data(self, game_status: Dict, conversation_log: str) -> Dict:
        """프롬프트 주입용 데이터 가공"""
        base_data = {
            "my_id": self.id,
            "my_role_name": self.inv_role_map.get(game_status.get("my_role")),
            "current_day": game_status.get("day", 1),
            "current_phase": game_status.get("phase"),
            "alive_players": game_status.get("alive_status", []),
            "last_execution_result": game_status.get("last_execution_result"),
            "last_night_result": game_status.get("last_night_result"),
            "mafia_members": game_status.get("mafia_team_members"),
            "police_investigations": game_status.get("police_investigation_results"),
            "vote_history": game_status.get("vote_records", []),
            "conversation_history": conversation_log,
            "belief_matrix_summary": self.belief.tolist(),
        }
        # Filter out keys where the value is None to prevent information leakage
        return {k: v for k, v in base_data.items() if v is not None}

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
