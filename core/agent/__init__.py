"""
Character module - now simplified to use only RationalCharacter
"""
import config
from core.characters.base import BaseCharacter
from core.characters.rational import RationalCharacter


def create_player(char_id: int, player_id: int) -> BaseCharacter:
    """
    Create a rational player (personality-based agents removed).
    char_id parameter is kept for backward compatibility but ignored.
    """
    # All players now use RationalCharacter
    return RationalCharacter(player_id)

