from world.base_world import BaseWorld
import gymnasium as gym
import numpy as np
import gym_maze


class Maze5x5World(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=50,
    ):
        super().__init__(gym_env_name)
        assert render_mode in [True, False]
        self.env = gym.make(gym_env_name, enable_render=render_mode)
        self.state_space = np.arange(25)
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length

    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state[1] * 5 + state[0]

    def step(self, action):
        self.steps += 1
        assert action in range(4)
        state, reward, done, truncated, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length or truncated:
            done = True

        return state[1] * 5 + state[0], reward, done

    def get_accu_reward(self):
        return self.accu_reward
