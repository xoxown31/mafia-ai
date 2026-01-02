"""
Character module
"""

import config
from core.agent.baseAgent import BaseAgent
from core.agent.llmAgent import LLMAgent


def create_player(player_id: int) -> BaseAgent:
    """
    Create a rational player (personality-based agents removed).
    """
    # All players now use RationalCharacter
    return LLMAgent(player_id)
