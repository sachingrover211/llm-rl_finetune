import gymnasium as gym
import numpy as np

from world.base_world import BaseWorld


class CartpoleWorld(BaseWorld):
    ACTIONS = {
        "left": 0,
        "right": 1
    }

    def __init__(self, _render_mode, is_continuous = True, num_pos_bins = 10, num_angle_bins = 10):
        super().__init__("CartPole-v1")
        self.render_mode = _render_mode
        self.is_continuous = is_continuous
        self.num_pos_bins = num_pos_bins
        self.num_angle_bins = num_angle_bins
        self.total_reward = 0
        if not self.is_continuous:
            self.position_bins = np.linspace(-2.4, 2.4, num_pos_bins)
            self.angle_bins = np.linspace(-0.2095, 0.2095, num_angle_bins)



    def reset(self):
        if self.render_mode in ["human", "rgb_array"]:
            self.env = gym.make(
                self.name,
                render_mode = self.render_mode
            )
        else:
            self.env = gym.make(self.name)

        self.state, _ = self.env.reset()
        self.total_reward = 0

        return self.decode_state(self.state)


    def reset_with_video(self, folder, _name):
        self.env = gym.make(
            self.name,
            render_mode =  "rgb_array"
        )

        self.env = gym.wrappers.RecordVideo(
            env = self.env,
            video_folder = folder,
            name_prefix = _name,
        )

        self.state, _ = self.env.reset()
        self.total_reward = 0

        return self.decode_state(self.state)


    def decode_state(self, state):
        X = state[0]
        theta = state[2]
        if self.is_continuous:
            V = state[1]
            angular_theta = state[3]
            return np.array([X, V, theta, angular_theta])
        else:
            X_idx = (
                np.digitize(X, self.position_bins) - self.num_pos_bins // 2
            )
            theta_idx = (
                np.digitize(theta, self.angle_bins) - self.num_angle_bins // 2
            )

            # only digitize X and theta for now
            # TODO: descritize velocity and angular velocity
            return np.array([X_idx, theta_idx])


    def step(self, action, is_recording = False):
        #action = CartpoleWorld.ACTIONS[action]
        state, reward, done, _, _ = self.env.step(action)
        if is_recording:
            self.env.render()

        self.total_reward += reward

        return self.decode_state(state), reward, done


    def get_accu_reward(self):
        return self.total_reward


    def start_video_recorder(self):
        self.env.start_video_recorder()


    def close_video_recorder():
        self.env.close_video_recorder()


    def close(self):
        self.env.close()
