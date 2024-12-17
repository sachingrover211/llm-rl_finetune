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
        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)

    def get_action(self, state):
        state = state.T
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
        self.weight = np.array(weight_and_bias_list[:-1])
        self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)
