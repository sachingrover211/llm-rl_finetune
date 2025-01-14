from world.base_world import BaseWorld
import gymnasium as gym
import numpy as np


class MujocoHopperWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=300,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length

    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state

    def step(self, action):
        self.steps += 1
        action = action[0]
        state, reward, done, _, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length:
            done = True

        return state, reward, done

    def get_accu_reward(self):
        return self.accu_reward
