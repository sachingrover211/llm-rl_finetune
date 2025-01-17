import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)

        self.dim_states =dim_states
        self.dim_actions = dim_actions

        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)

    def initialize_policy(self):
        self.weight = np.round((np.random.rand(self.dim_states, self.dim_actions) - 0.5) * 12, 1)
        self.bias = np.round(np.random.rand(1, self.dim_actions) - 0.5 * 12, 1)

    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        output = "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        output += "Bias:\n"
        for b in self.bias:
            output += ", ".join([str(i) for i in b])
            output += "\n"

        return output

    def update_policy(self, weight_and_bias_list):
        if weight_and_bias_list is None:
            return
        self.weight = np.array(weight_and_bias_list[:-1])
        self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)
