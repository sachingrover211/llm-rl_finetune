import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

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
        self.render_mode = _render_mode


    def reset(self):
        if self.render_mode in ["human", "rgb_array"]:
            self.env = gym.make(
                self.name,
                desc = self.grid,
                render_mode = self.render_mode
            )
        else:
            self.env = gym.make(
                self.name,
                desc = self.grid
            )
        self.state, _ = self.env.reset()

        return self.decode_state(self.state)


    def decode_state(self, state):
        col = state % self.grid_size
        row = int(state / self.grid_size)

        return (row, col)


    def step(self, action):
        action = FrozenLakeWorld.ACTIONS[action]
        state, reward, done, truncated, _ = self.env.step(action)

        return self.decode_state(state), reward, done, truncated


    def save_domain(self, logdir, index = 0):
        _env = gym.make(
            self.name,
            desc = self.grid,
            render_mode = "rgb_array"
        )
        _, _ = _env.reset()
        img = env.render()
        plt.imshow(img)
        plt.savefig(f"{logdir}/frozen_lake_{index}.png")
        _env.close()

