from world.base_world import BaseWorld
import gymnasium as gym
import numpy as np


class PendulumWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        num_costheta_bins=5,
        num_sintheta_bins=5,
        num_angular_velocity_bins=5,
        num_action_bins=5,
        max_traj_length=200,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.num_costheta_bins = num_costheta_bins
        self.num_sintheta_bins = num_sintheta_bins
        self.num_angular_velocity_bins = num_angular_velocity_bins
        self.num_action_bins = num_action_bins
        self.costheta_bins = np.linspace(-1.01, 1.01, num_costheta_bins)
        self.sintheta_bins = np.linspace(-1.01, 1.01, num_sintheta_bins)
        self.angular_velocity_bins = np.linspace(-8.01, 8.01, num_angular_velocity_bins)
        self.action_bins = np.linspace(-2, 2, num_action_bins)
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
        action = self.get_action(action)
        state, reward, done, _, _ = self.env.step([action])
        self.accu_reward += reward

        if self.steps >= self.max_traj_length:
            done = True

        return self.discretize_state(state), reward, done

    def discretize_state(self, state):
        costheta_idx = (
            np.digitize(state[0], self.costheta_bins) - self.num_costheta_bins // 2
        )
        sintheta_idx = (
            np.digitize(state[1], self.sintheta_bins) - self.num_sintheta_bins // 2
        )
        angular_velocity_idx = (
            np.digitize(state[2], self.angular_velocity_bins)
            - self.num_angular_velocity_bins // 2
        )
        return (costheta_idx, sintheta_idx, angular_velocity_idx)
    
    def discretize_action(self, action):
        return np.digitize(action, self.action_bins) - self.num_action_bins // 2
    
    def get_action(self, action_idx):
        return self.action_bins[action_idx] + 0.2

    def get_accu_reward(self):
        return self.accu_reward
