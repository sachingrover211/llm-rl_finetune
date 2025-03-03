from world.base_world import BaseWorld
import gymnasium as gym
import numpy as np


class MountainCarWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        num_position_bins=10,
        num_velocity_bins=20,
        max_traj_length=300,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.position_bins = np.linspace(-1.2, 0.6, num_position_bins)
        self.velocity_bins = np.linspace(-0.07, 0.07, num_velocity_bins)
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length

    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return self.discretize_state(state)

    def step(self, action):
        self.steps += 1
        action = action[0]
        assert action in range(3)
        state, reward, done, _, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length:
            done = True

        return self.discretize_state(state), reward, done

    def discretize_state(self, state):
        position_idx = (
            np.digitize(state[0], self.position_bins) - self.num_position_bins // 2
        )
        velocity_idx = (
            np.digitize(state[1], self.velocity_bins) - self.num_velocity_bins // 2
        )
        return (position_idx, velocity_idx)

    def get_accu_reward(self):
        return self.accu_reward


# There was a lot of machinery added to make sure that descritization can work
# for mountain car. Not to support unnecessary methods, inheriting it from
# BaseWorld. Ideally it should have inheritted from the MountainCarWorld
class MountainCarContinuousWorld(BaseWorld):
    def __init__(self, gym_env_name, render_mode, max_traj_length=1000):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.max_traj_length = max_traj_length


    def reset(self):
        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state


    def step(self, action):
        self.steps += 1
        state, reward, done, truncated, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length:
            done = True


        return state, reward, done or truncated


    def get_accu_reward(self):
        return self.accu_reward
