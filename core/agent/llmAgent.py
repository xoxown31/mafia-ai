import os
import json
import yaml
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

        # Load prompts
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

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

    def _create_prompt_and_call(self, phase_key: str, **kwargs) -> str:
        """Common method to generate prompt and call LLM"""
        prompt_data = self.prompts.get(phase_key)
        if not prompt_data:
            return json.dumps({"error": f"Prompt for {phase_key} not found"})
            
        system_msg = prompt_data['system']
        user_template = prompt_data['user']
        
        user_msg = user_template.format(**kwargs)
        
        return self._call_llm(system_msg, user_msg)

    def _handle_discussion(self, conversation_log: str) -> str:
        """낮 토론: 신념 분석 -> 전략 수립 -> 대사 생성"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        
        response_json = self._create_prompt_and_call(
            "DISCUSSION",
            id=self.id,
            role_name=self.role.name,
            status_json=status_json,
            conversation_log=conversation_log,
            belief_matrix=self.belief.tolist()
        )
        self._apply_llm_belief_updates(response_json)
        return response_json

    def _handle_vote(self, conversation_log: str) -> str:
        """낮 투표: 최적의 처형 대상 선정"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        return self._create_prompt_and_call(
            "VOTE",
            role_name=self.role.name,
            status_json=status_json,
            conversation_log=conversation_log
        )

    def _handle_final_vote(self, conversation_log: str) -> str:
        """사형 찬반 투표"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        return self._create_prompt_and_call(
            "FINAL_VOTE",
            role_name=self.role.name,
            status_json=status_json
        )

    def _handle_night(self, conversation_log: str) -> str:
        """밤 행동: 특수 능력 수행"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        return self._create_prompt_and_call(
            "NIGHT",
            role_name=self.role.name,
            id=self.id,
            status_json=status_json
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Upstage API 호출 및 예외 처리"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
