import os
import json
import yaml
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING, Any
from openai import OpenAI
from dotenv import load_dotenv

from core.agent.baseAgent import BaseAgent
from config import config, Role, Phase, EventType, ActionType
from state import GameStatus, GameEvent, GameAction

if TYPE_CHECKING:
    from core.logger import LogManager

load_dotenv()


class LLMAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        role: Role = Role.CITIZEN,
        logger: Optional["LogManager"] = None,
    ):
        super().__init__(player_id, role)

        # LogManager 인스턴스 (내러티브 해석용)
        self.logger = logger

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

    def _phase_to_korean(self, phase: Phase) -> str:
        """Phase enum을 한글로 변환"""
        phase_map = {
            Phase.GAME_START: "게임 시작",
            Phase.DAY_DISCUSSION: "낮 토론",
            Phase.DAY_VOTE: "투표",
            Phase.DAY_EXECUTE: "처형 여부 결정",
            Phase.NIGHT: "밤",
        }
        return phase_map.get(phase, phase.name)

    def _group_events_by_day_phase(self, events: List[GameEvent]) -> dict:
        """이벤트를 day와 phase로 그룹화"""
        from collections import defaultdict

        grouped = defaultdict(list)
        for event in events:
            key = (event.day, event.phase)
            grouped[key].append(event)
        return dict(sorted(grouped.items()))

    def _format_event(self, event: GameEvent) -> str:
        """GameEvent를 자연어 문장으로 변환"""
        event_type = event.event_type
        actor = f"Player {event.actor_id}" if event.actor_id != -1 else "시스템"
        target = f"Player {event.target_id}" if event.target_id != -1 else "없음"

        if event_type == EventType.CLAIM:
            role_name = (
                Role(event.value).name if event.value is not None else "알 수 없음"
            )
            if event.actor_id == event.target_id:
                return f"{actor}가 자신이 {role_name}이라고 주장했습니다."
            else:
                return f"{actor}가 {target}을(를) {role_name}(이)라고 주장했습니다."

        elif event_type == EventType.VOTE:
            return f"{actor}가 {target}에게 투표했습니다."

        elif event_type == EventType.EXECUTE:
            role_name = (
                Role(event.value).name if event.value is not None else "알 수 없음"
            )
            return f"{target}({role_name})이(가) 처형되었습니다."

        elif event_type == EventType.KILL:
            return f"{actor}가 {target}을(를) 살해 대상으로 지목했습니다."

        elif event_type == EventType.PROTECT:
            return f"{actor}가 {target}을(를) 보호했습니다."

        elif event_type == EventType.POLICE_RESULT:
            role_name = (
                Role(event.value).name if event.value is not None else "알 수 없음"
            )
            return f"{actor}가 {target}을(를) 조사한 결과 {role_name}입니다."

        else:
            return f"알 수 없는 이벤트: {event_type.name}"

    def _format_game_status(self) -> str:
        """
        GameStatus를 LLM이 이해하기 쉬운 형태로 변환합니다.

        Returns:
            자연어 형식의 게임 상태 설명
        """
        status = self.current_status
        lines = []

        # 기본 정보
        lines.append(f"### 현재 게임 상태")
        lines.append(f"- Day {status.day}, {self._phase_to_korean(status.phase)}")
        lines.append("")

        # 생존자 목록
        alive_players = [p.id for p in status.players if p.alive]
        dead_players = [p.id for p in status.players if not p.alive]
        lines.append(
            f"### 생존자: {', '.join(f'Player {pid}' for pid in alive_players)}"
        )
        if dead_players:
            lines.append(
                f"### 사망자: {', '.join(f'Player {pid}' for pid in dead_players)}"
            )
        lines.append("")

        # 이벤트 히스토리 (자연어 변환)
        lines.append("### 게임 이벤트 기록")
        if not status.action_history:
            lines.append("아직 이벤트가 없습니다.")
        else:
            grouped_events = self._group_events_by_day_phase(status.action_history)
            for (day, phase), events in grouped_events.items():
                lines.append(f"\n**Day {day}, {self._phase_to_korean(phase)}:**")
                for event in events:
                    lines.append(f"  - {self._format_event(event)}")

        return "\n".join(lines)

    def update_belief(self, history: List[GameEvent]):
        # 1. Hard-coded Update (Fact)
        for event in history:
            if event.event_type in [EventType.POLICE_RESULT, EventType.SYSTEM_MESSAGE]:
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
            response_data = self._execute_ai_logic("BELIEF_UPDATE")
            try:
                # _execute_ai_logic이 이미 딕셔너리를 반환하므로 json.loads 불필요
                updates = response_data.get("belief_updates", [])
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
            except (AttributeError, KeyError) as e:
                print(
                    f"[Player {self.id}] Belief update failed: {e}\nResponse from LLM: {response_data}"
                )

    def translate_to_engine(self, action_dict: Dict[str, Any]) -> GameAction:
        """
        LLM JSON 응답을 MafiaAction으로 변환하는 순수 번역기

        딕셔너리 구조:
        - target_id: int (지목 대상, -1은 없음)
        - role: Optional[Role] (주장하는 역할)
        - discussion_status: str (토론 종료 여부, "End"면 PASS)

        Args:
            action_dict: LLM이 반환한 JSON 딕셔너리

        Returns:
            MafiaAction 객체
        """
        target_id = action_dict.get("target_id", -1)
        # target_id가 None인 경우 -1로 처리
        if target_id is None:
            target_id = -1
        role_str = action_dict.get("role")
        discussion_status = action_dict.get("discussion_status", "Continue")

        # 역할을 Role enum으로 변환 (정수 또는 문자열 처리)
        claim_role = None
        if role_str is not None:
            if isinstance(role_str, int):
                # 정수인 경우 (0:CITIZEN, 1:POLICE, 2:DOCTOR, 3:MAFIA)
                try:
                    claim_role = Role(role_str)
                except ValueError:
                    print(f"[Player {self.id}] Invalid role integer: {role_str}")
            elif isinstance(role_str, str):
                # 문자열인 경우
                role_upper = role_str.upper()
                if hasattr(Role, role_upper):
                    claim_role = Role[role_upper]
                else:
                    print(f"[Player {self.id}] Invalid role string: {role_str}")

        # 토론 종료 → PASS
        if discussion_status == "End":
            return GameAction(
                action_type=ActionType.PASS, target_id=-1, claim_role=None
            )

        # 역할 주장 (타겟 포함 여부 무관하게 CLAIM으로 통합)
        if claim_role is not None:
            return GameAction(
                action_type=ActionType.CLAIM, target_id=target_id, claim_role=claim_role
            )

        # 단순 지목
        if target_id != -1:
            return GameAction(
                action_type=ActionType.TARGET_ACTION,
                target_id=target_id,
                claim_role=None,
            )

        # 기권
        return GameAction(action_type=ActionType.PASS, target_id=-1, claim_role=None)

    def get_action(self) -> GameAction:
        """
        LLM의 JSON 응답을 MafiaAction으로 변환하여 반환

        Note: 이 메서드는 내부적으로 _execute_ai_logic()을 호출하여
        LLM 응답을 얻은 후 translate_to_engine()으로 변환합니다.
        """
        if not self.current_status:
            return GameAction(
                action_type=ActionType.PASS, target_id=-1, claim_role=None
            )

        phase_name = self.current_status.phase.name
        role_name = self.role.name

        role_specific_key = f"{role_name}_{phase_name}"
        if role_specific_key in self.yaml_data:
            prompt_key = role_specific_key
        else:
            prompt_key = phase_name

        # LLM 실행 및 JSON 응답 파싱
        action_dict = self._execute_ai_logic(prompt_key)

        # translate_to_engine()을 통한 변환
        return self.translate_to_engine(action_dict)

    def _execute_ai_logic(self, prompt_key: str) -> Dict[str, Any]:
        """LLM 실행 및 응답을 딕셔너리로 파싱하여 반환"""
        status_json = self.current_status.model_dump_json(exclude_none=True)
        phase_name = self.current_status.phase.name

        prompt_data = self.yaml_data.get(prompt_key)
        if not prompt_data:
            print(
                f"[Player {self.id}] Warning: Prompt key '{prompt_key}' not found, returning empty action"
            )
            # 프롬프트가 없으면 빈 target_id 반환 (PASS로 처리됨)
            return {"target_id": -1}

        role_specific_system_msg = prompt_data.get("system", "")
        json_schema = self.phase_schemas.get(
            prompt_key, self.phase_schemas.get(phase_name, {})
        )
        json_instruction = self.json_format_block.format(json_schema=json_schema)
        final_system_msg = f"{role_specific_system_msg}\n{json_instruction}"

        user_template = prompt_data.get("user", "")
        conversation_log = self._create_conversation_log()

        # 신뢰도 행렬을 마크다운 테이블로 변환
        belief_markdown = self._belief_to_markdown()

        # 게임 상태를 읽기 쉬운 형태로 변환
        formatted_status = self._format_game_status()

        game_data = self.game_data_block.format(
            role_name=self.role.name,
            id=self.id,
            game_status=formatted_status,
            belief_matrix=belief_markdown,
            conversation_log=conversation_log,
        )
        print(f"[Player {self.id}] Game Data for LLM:\n{game_data}\n")
        final_user_msg = user_template.format(game_data=game_data)
        print(
            f"[Player {self.id}] Final System Msg:\n{final_system_msg}\n Final User Msg:\n{final_user_msg}\n"
        )

        # LLM 호출 및 JSON 파싱
        response_str = self._call_llm(final_system_msg, final_user_msg)
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"[Player {self.id}] Failed to parse LLM response: {e}")
            return {
                "error": "Invalid JSON response from LLM",
                "raw_response": response_str,
            }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM API 호출 - JSON 문자열 반환"""
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
        """
        대화 기록을 생성합니다.
        LogManager의 interpret_event()를 사용하여 일관된 내러티브를 생성합니다.
        """
        if self.current_status.phase != Phase.DAY_DISCUSSION:
            return "토론 단계가 아님"

        if not self.logger:
            return "LogManager가 설정되지 않아 기록을 표시할 수 없습니다."

        # LogManager를 통한 통합 해석 (중복 제거)
        log_lines = [
            self.logger.interpret_event(e)
            for e in self.current_status.action_history
            if e.event_type == EventType.CLAIM
        ]

        return "\n".join(log_lines) if log_lines else "아직 아무도 주장하지 않았습니다."

    def _belief_to_markdown(self) -> str:
        """
        신뢰도 행렬을 마크다운 테이블 형식으로 변환합니다.
        LLM이 더 쉽게 이해할 수 있도록 구조화된 형식을 제공합니다.

        Returns:
            마크다운 형식의 신뢰도 테이블
        """
        # 헤더 생성
        headers = ["Player ID", "시민", "경찰", "의사", "마피아"]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("|" + "---|" * len(headers))

        # 각 플레이어의 신뢰도 값
        for player_id in range(config.game.PLAYER_COUNT):
            row = [f"Player {player_id}"]
            for role_idx in range(len(Role)):
                belief_value = self.belief[player_id, role_idx]
                row.append(f"{belief_value:.1f}")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
