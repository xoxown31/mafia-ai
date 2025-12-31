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
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f)

        self.phase_schemas = self.yaml_data.get("PHASE_SCHEMAS", {})
        self.json_format_block = self.yaml_data.get("JSON_FORMAT_BLOCK", "")
        self.game_data_block = self.yaml_data.get("GAME_DATA_BLOCK", "")

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
            except (json.JSONDecodeError, AttributeError) as e:
                print(
                    f"[Player {self.id}] Belief update failed: {e}\nResponse from LLM: {response_json}"
                )

    def get_action(self) -> str:
        if not self.current_status:
            return json.dumps({"error": "No game status observed"})

        phase_name = self.current_status.phase.name
        role_name = self.role.name

        role_specific_key = f"{role_name}_{phase_name}"
        if role_specific_key in self.yaml_data:
            prompt_key = role_specific_key
        else:
            prompt_key = phase_name

        return self._execute_ai_logic(prompt_key)

    def _execute_ai_logic(self, prompt_key: str) -> str:
        status_json = self.current_status.model_dump_json(exclude_none=True)
        phase_name = self.current_status.phase.name

        prompt_data = self.yaml_data.get(prompt_key)
        if not prompt_data:
            return json.dumps({"error": f"Prompt for {prompt_key} not found"})

        role_specific_system_msg = prompt_data.get("system", "")
        json_schema = self.phase_schemas.get(
            prompt_key, self.phase_schemas.get(phase_name, {})
        )
        json_instruction = self.json_format_block.format(json_schema=json_schema)
        final_system_msg = f"{role_specific_system_msg}\n{json_instruction}"

        user_template = prompt_data.get("user", "")
        conversation_log = self._create_conversation_log()
        game_data = self.game_data_block.format(
            role_name=self.role.name,
            id=self.id,
            status_json=status_json,
            belief_matrix=self.belief.tolist(),
            conversation_log=conversation_log,
        )
        print(f"[Player {self.id}] Game Data for LLM:\n{game_data}\n")
        final_user_msg = user_template.format(game_data=game_data)
        print(
            f"[Player {self.id}] Final System Msg:\n{final_system_msg}\n Final User Msg:\n{final_user_msg}\n"
        )
        return self._call_llm(final_system_msg, final_user_msg)

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

    def _create_conversation_log(self) -> str:
        if self.current_status.phase != Phase.DAY_DISCUSSION:
            return "토론 단계가 아님"
        phase_name = self.current_status.phase.name
        day = self.current_status.day
        log_lines = []

        action_string = ""

        for event in self.current_status.action_history:
            if event.event_type != EventType.CLAIM:
                continue
            actor_id = event.actor_id
            target_id = event.target_id if event.target_id is not None else -1
            claimed_role = event.value if isinstance(event.value, Role) else None
            if claimed_role is not None:
                if target_id == actor_id or target_id == -1:
                    action_string = (
                        f"Player {actor_id}는 자신이 {claimed_role.name}라고 주장"
                    )
                else:
                    action_string = f"Player {actor_id}는 Player {target_id}가 {claimed_role.name}라고 주장"
            else:
                action_string = f"Player {actor_id}가 침묵."
            log_lines.append(f"{day}일 {phase_name} | " + action_string)

        if not log_lines:
            return "아직 아무도 주장하지 않았습니다."
        else:
            return "\n".join(log_lines)
