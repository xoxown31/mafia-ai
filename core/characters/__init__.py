import config
from core.characters.base import BaseCharacter
from core.characters.orator import Orator
from core.characters.follower import Follower
from core.characters.grudger import Grudger
from core.characters.analyst import Analyst
from core.characters.maverick import Maverick
from core.characters.copycat import CopyCat
from core.characters.copykitten import CopyKitten


def create_player(char_id: int, player_id: int) -> BaseCharacter:
    """ID에 맞는 성격 클래스를 찾아 인스턴스(Player) 생성"""
    if char_id == config.CHAR_COPYCAT:
        return CopyCat(player_id)
    elif char_id == config.CHAR_GRUDGER:
        return Grudger(player_id)
    elif char_id == config.CHAR_COPYKITTEN:
        return CopyKitten(player_id)
    elif char_id == config.CHAR_ORATOR:
        return Orator(player_id)
    elif char_id == config.CHAR_FOLLOWER:
        return Follower(player_id)
    elif char_id == config.CHAR_ANALYST:
        return Analyst(player_id)
    elif char_id == config.CHAR_MAVERICK:
        return Maverick(player_id)
    else:
        # 기본값으로 Orator 사용
        return Orator(player_id)
