import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from world.base_world import BaseWorld


class FrozenLakeWorld(BaseWorld):
    ACTIONS = {
        "left": 0,
        "down": 1,
        "right": 2,
        "up": 3
    }

    def __init__(self, _render_mode, _size: int = 4):
        super().__init__("FrozenLake-v1")
        self.grid_size = _size
        self.grid = generate_random_map(self.grid_size)
        self.env = gym.make(
            self.name,
            desc = self.grid
        )



    def reset(self):
        self.state, _ = self.env.reset()

        return self.decode_state(self.state)


    def decode_state(self, state):
        col = state % self.grid_size
        row = int(state / self.grid_size)

        return (row, col)


    def step(self, action):
        action = FrozenLakeWorld(action)
        state, reward, done, _, _ = self.env.step(action)

        return self.decode_state(state), reward, done
