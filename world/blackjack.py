from world.base_world import BaseWorld
import gymnasium as gym
import numpy as np


class BlackjackWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=50,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.player_current_sum = np.arange(21)
        self.dealer_showing_card_value = np.arange(11)
        self.usable_ace = np.arange(2)
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
        action = int(np.reshape(action, (1,)))
        assert action in range(2)
        state, reward, done, truncacted, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length or truncacted:
            done = True

        return state, reward, done

    def get_accu_reward(self):
        return self.accu_reward
