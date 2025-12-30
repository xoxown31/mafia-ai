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
        # 1. Hard-coded Update (Fact)
        for event in history:
            if event.event_type == EventType.POLICE_RESULT:
                target_id = event.target_id
                role_found = event.value
                if target_id is not None and isinstance(role_found, Role):
                    role_idx = int(role_found)
                    self.belief[target_id, role_idx] = 100.0
                    for r in Role:
                        if r != role_idx:
                            self.belief[target_id, r] = -100.0

        # 2. LLM-based Inference (Hunch)
        if self.current_status:
            response_json = self._execute_ai_logic("BELIEF_UPDATE")
            try:
                data = json.loads(response_json)
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

    def get_action(self) -> str:
        if not self.current_status:
            return json.dumps({"error": "No game status observed"})

        phase_name = self.current_status.phase.name
        role_name = self.role.name
        
        prompt_key = f"{role_name}_{phase_name}"
        if prompt_key not in self.prompts:
            prompt_key = phase_name

        return self._execute_ai_logic(prompt_key)

    def _execute_ai_logic(self, prompt_key: str) -> str:
        status_json = self.current_status.model_dump_json(exclude_none=True)
        
        prompt_data = self.prompts.get(prompt_key)
        if not prompt_data:
            return json.dumps({"error": f"Prompt for {prompt_key} not found"})

        system_msg = prompt_data['system']
        user_template = prompt_data['user']
        
        user_msg = user_template.format(
            id=self.id,
            role_name=self.role.name,
            status_json=status_json,
            belief_matrix=self.belief.tolist()
        )

        return self._call_llm(system_msg, user_msg)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
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
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})
