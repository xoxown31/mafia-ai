from core.engine.state import GameEvent


class LogEvent(GameEvent):
    """state.py 수정 없이 episode 필드를 인식하기 위한 확장 클래스"""

    episode: int = 1
