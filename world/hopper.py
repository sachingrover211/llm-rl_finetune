from world.base_world import BaseWorld


class HopperWorld(BaseWorld):
    def __init__(self, _render_mode):
        super().__init__("Hopper-v5")
        self.render_mode = _render_mode


