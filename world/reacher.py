import numpy as np
from world.base_world import BaseWorld


class ReacherWorld(BaseWorld):
    def __init__(self, _render_mode):
        super().__init__("Reacher-v5")
        self.render_mode = _render_mode


    def step(self, action):
        action = action.reshape(-1)
        state, reward, done, truncated, _ = self.env.step(action)
        self.total_reward += reward

        return self.decode_state(state), reward, done or truncated
