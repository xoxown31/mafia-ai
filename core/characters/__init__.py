import config
from core.characters.base import BaseCharacter
from core.characters.rational import RationalCharacter
from core.characters.copycat import CopyCat
from core.characters.grudger import Grudger
from core.characters.copykitten import CopyKitten


def create_player(char_id: int, player_id: int) -> BaseCharacter:
    """ID에 맞는 성격 클래스를 찾아 인스턴스(Player) 생성"""
    if char_id == config.CHAR_COPYCAT:
        return CopyCat(player_id)
    elif char_id == config.CHAR_GRUDGER:
        return Grudger(player_id)
    elif char_id == config.CHAR_COPYKITTEN:
        return CopyKitten(player_id)
    else:
        return RationalCharacter(player_id)
