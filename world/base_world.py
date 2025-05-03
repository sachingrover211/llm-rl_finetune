import gymnasium as gym


class BaseWorld:
    def __init__(self, name):
        self.name = name


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


    def interpolate_state(self, state):
        raise NotImplementedError("interpolate_state method not implemented")


    def discretize_state(self, state):
        raise NotImplementedError("discretize_state method not implemented")


    def interpolate_action(self, action):
        raise NotImplementedError("interpolate_action method not implemented")


    def discretize_action(self, action):
        raise NotImplementedError("discretize_action method not implemented")


    def decode_state(self, state):
        # when there is no change needed for the state
        return state


    def step(self, action):
        state, reward, done, truncated, _ = self.env.step(action)
        self.total_reward += reward

        return self.decode_state(state), reward, done or truncated


    def close(self):
        self.env.close()


    def get_accu_reward(self):
        return self.total_reward
