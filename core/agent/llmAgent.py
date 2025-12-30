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
        # 부모 클래스(BaseAgent) 초기화: self.id, self.role, self.belief 행렬 등 생성
        super().__init__(player_id, role)

        # API 설정
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-mini"

    def update_belief(self, game_status: Dict):
        """
        BaseAgent의 추상 메서드 구현.
        여기서는 게임 엔진에서 전달되는 고정된 사건(누가 죽었는지 등)에 대한
        기초적인 수치 업데이트를 수행할 수 있습니다.
        (상세한 추론 업데이트는 get_action 내 _apply_belief_updates에서 처리)

        """
        pass

    def get_action(self, game_status: Dict, conversation_log: str) -> str:
        """
        [핵심 함수] 상황(Phase)별로 분기하여 적절한 프롬프트를 실행하고 결과를 반환합니다.
        """
        phase = game_status.get("phase")

        # 1. 상황별 핸들러 호출
        if phase == config.PHASE_DAY_DISCUSSION:
            return self._handle_discussion(game_status, conversation_log)
        elif phase == config.PHASE_DAY_VOTE:
            return self._handle_vote(game_status, conversation_log)
        elif phase == config.PHASE_DAY_EXECUTE:
            return self._handle_final_vote(game_status, conversation_log)
        elif phase == config.PHASE_NIGHT:
            return self._handle_night(game_status, conversation_log)

        return json.dumps({"error": "Unknown phase"})

    # --- 내부 핸들러 메서드 ---

    def _handle_discussion(self, game_status: Dict, conversation_log: str) -> str:
        """낮 토론: 신념 업데이트 + 전략 수립 + 대사 생성"""
        input_data = self._prepare_input_data(game_status, conversation_log)
        prompt = f"""
        ### SYSTEM ROLE
        당신은 냉철한 마피아 게임 전략가입니다. 현재 단계는 [낮 토론]입니다.
        제공된 [GAME STATE]를 바탕으로 다른 플레이어에 대한 신념을 업데이트하고, 설득력 있는 대화를 생성하십시오.

        ### [GAME STATE]
        {input_data}

        ### TASK 1: Belief Update
        각 플레이어의 정체성에 대한 변화량($\Delta$)을 -100에서 100 사이의 정수로 계산하십시오.
        - 논리적 모순이나 수상한 발언을 하면 마피아 확률($\Delta$) 증가.
        - 본인({self.id})은 제외.

        ### TASK 2: Action Strategy & Speech
        최종 결정 수식 $L_i = S_{{belief, i}} + Bias_i$를 바탕으로 전략적 편향($Bias$)을 설정하고 대사를 작성하십시오.
        - 시민 팀: 의심자에게 양수의 $Bias$ 부여.
        - 마피아 팀: 무고한 시민을 몰아가거나 동료를 방어하는 기만 전략 사용.
        - Speech: 실제 채팅처럼 자연스러운 한국어로 작성하십시오. (예: "3번님, 아까 말씀이 좀 이상하신데요?")

        ### Task 3:
        상황이 명확하지 않을 때 토론을 자제할 확률을 조절하기 위해
        침묵 지수(silence_bias)를 0에서 100 사이의 실수로 설정하십시오.
        설정한 침묵 지수만큼 발언을 자제할 확률이 증가합니다.
        침묵할 경우에 silence 필드를 true로 설정하십시오.

        ### Task 4:
        토론을 계속 진행할지(Proceed) 아니면 종료할지(End) 결정하십시오.
        - 토론 종료 시점 : 모든 사람이 침묵하였을 때에 discussion_status를 "End"로 설정하십시오.

        ### Task 5: 
        토론에서는 3가지 주장만 가능 (본인의 직업, 타인의 직업, 침묵)
        - 본인의 직업 주장 시 : 본인은 시민(0)/경찰(1)/의사(2)라고 주장 및 role 필드에 주장한 직업 기록
        - 타인의 직업 주장 시 : 특정 플레이어가 경찰(1)/의사(2)/마피아(3)/라고 주장 
          및 target_id 필드에 대상 id 기록 및 role 필드에 주장한 직업 기록
        - 침묵 시 : 아무 말도 하지 않음
        - 각 주장한 것을 claim 필드에 기록 (본인의 직업: 0, 타인의 직업: 1, 침묵: 2)

        ### OUTPUT FORMAT (JSON)
        {{
            "belief_updates": [{{
                "player_id": int,
                "role": "CITIZEN" or "POLICE" or "DOCTOR" or "MAFIA",
                "delta": int (-100 to 100)
            }}],
            "action_strategy": {{
                "biases": [float, float, float, float, float, float, float, float],
                "silence_bias": float,
                "strategy_note": "전략 요약"
            }},
            "speech": "실제 내뱉을 한국어 문장",
            "discussion_status": "Proceed" or "End",
            "silence" : bool,
            "claim": int,
            "target_id": Optional[int],
            "role": Optional[int],
        }}
        """
        response_json = self._call_llm(prompt)
        self._apply_belief_updates(response_json)  # LLM의 분석을 내 신념 행렬에 반영
        return response_json

    def _handle_vote(self, game_status: Dict, conversation_log: str) -> str:
        """낮 투표: 최적의 처형 대상 선정"""
        input_data = self._prepare_input_data(game_status, conversation_log)

        prompt = f"""
        ### TASK: DAY VOTE
        토론이 종료되었습니다. 당신은 [{input_data['my_role_name']}]로서 누구를 처형할지 결정해야 합니다.
        [GAME STATE]: {input_data}

        가장 마피아로 의심되는 생존자 1명을 선택하십시오. 
        만약 본인이 마피아라면, 팀의 승리를 위해 무고한 시민을 타겟팅하십시오.
        상황이 확실 시 않은 경우에는 투표를 기권할 수 있습니다 (target_id에 -1).

        ### OUTPUT FORMAT (JSON)
        {{
            "target_id": int,
            "reason": "해당 플레이어를 투표한 결정적 이유"
        }}
        """
        return self._call_llm(prompt)

    def _handle_final_vote(self, game_status: Dict, conversation_log: str) -> str:
        """사형 찬반 투표: 최종 결정 내리기"""
        input_data = self._prepare_input_data(game_status, conversation_log)
        input_data["excute_target"] = game_status.get("execution_target")

        prompt = f"""
        ### TASK: DAY VOTE
        토론이 종료되었습니다. 당신은 [{input_data['my_role_name']}]로서 현재 {game_status.get("execution_target")}을 죽일지 살릴지 선택해야 합니다.
        [GAME STATE]: {input_data}

        신뢰도를 기반으로 
        처형에 찬성하면 1, 반대하면 -1, 기권하면 0을 선택하여 주세요.

        ### OUTPUT FORMAT (JSON)
        {{
            agree_execution: int,
        }}
        """
        return self._call_llm(prompt)

    def _handle_night(self, game_status: Dict, conversation_log: str) -> str:
        """밤 행동: 직업별 특수 능력 수행"""
        input_data = self._prepare_input_data(game_status, conversation_log)
        role_name = input_data["my_role_name"]

        prompt = f"""
        ### TASK: NIGHT ACTION
        당신은 [{role_name}]입니다. 밤의 행동을 결정하십시오.
        [GAME STATE]: {input_data}

        ### STRATEGY GUIDE
        - 마피아: 경찰/의사로 의심되는 핵심 시민 제거. (동료 지목 금지)
        - 경찰: 마피아 의심자 조사. (중복 조사 금지)
        - 의사: 공격받을 것 같은 인물(본인 포함) 보호.

        ### 신뢰도 기반 행동 결정
        각 플레이어에 대한 신뢰도 점수를 분석하여 최적의 행동 대상을 선택하십시오.
        경찰의 조사 대상이나 의사의 보호 대상도 신뢰도에 기반하여 결정하십시오.
        경찰의 조사 결과를 바탕으로 신념 행렬을 업데이트하는 것도 잊지 마십시오.

        ### OUTPUT FORMAT (JSON)
        {{
            "belief_updates": [{{
                "player_id": int,
                "role": "CITIZEN" or "POLICE" or "DOCTOR" or "MAFIA",
                "delta": int (-100 to 100)
            }}],
            "target_id": int,
        }}
        """
        response_json = self._call_llm(prompt)
        self._apply_belief_updates(response_json)
        return self._call_llm(prompt)

    # --- 유틸리티 및 헬퍼 메서드 ---

    def _prepare_input_data(self, game_status: Dict, conversation_log) -> Dict:
        """프롬프트에 주입할 공통 데이터 구조 생성"""
        return {
            "my_id": self.id,
            "my_role_name": self._get_role_name(self.role),
            "current_day": game_status.get("day", 1),
            "alive_players": game_status.get("alive_status", []),
            "last_execution": game_status.get("last_execution_result"),
            "last_night_kill": game_status.get("last_night_kill"),
            "mafia_team": game_status.get("mafia_team_members"),
            "police_results": game_status.get("police_investigation_results"),
            "doctor_protections": game_status.get("doctor_protection_results"),
            "belief_matrix": self.belief.tolist(),
            "conversation_history": conversation_log,
        }

    def _call_llm(self, prompt: str) -> str:
        """Upstage API 호출 공통 로직"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 이모지를 사용하지 않는 냉철한 마피아 게임 전략가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


    def _apply_belief_updates(self, raw_json: str):
        """LLM의 분석 결과를 내부 신념 행렬(N x 4)에 업데이트"""
        try:
            data = json.loads(raw_json)
            updates = data.get("belief_updates", [])
            role_map = {"CITIZEN": 0, "POLICE": 1, "DOCTOR": 2, "MAFIA": 3}

            for up in updates:
                p_id = up["player_id"]
                role_key = up["role"]
                delta = up["delta"]

                if 0 <= p_id < config.PLAYER_COUNT and role_key in role_map:
                    col_idx = role_map[role_key]
                    # numpy 행렬 업데이트 (0~100 사이로 클리핑)
                    new_val = self.belief[p_id, col_idx] + delta
                    self.belief[p_id, col_idx] = np.clip(new_val, -100, 100)
        except Exception as e:
            pass  # 로그 출력 생략 (실제 환경에서는 로깅 권장)

    def _get_role_name(self, role_int: int) -> str:
        mapping = {0: "시민", 1: "마피아", 2: "경찰", 3: "의사"}
        return mapping.get(role_int, "알 수 없음")
