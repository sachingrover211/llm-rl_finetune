import world.pong_env
from world.base_world import BaseWorld


class PongWorld(BaseWorld):
    def __init__(self, _render_mode):
        super().__init__("Pong-v0")
        self.render_mode = _render_mode
