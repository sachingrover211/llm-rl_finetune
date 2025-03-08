import numpy as np
from agent.policy.base_policy import Policy


class MLPPolicy(Policy):
    def __init__(self, dim_states, dim_l1, dim_actions):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_l1 = dim_l1
        self.dim_actions = dim_actions

        self.weight_l1 = np.random.rand(self.dim_states, self.dim_l1)
        self.bias_l1 = np.random.rand(1, self.dim_l1)
        self.weight = np.random.rand(self.dim_l1, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)

    def initialize_policy(self):
        self.weight_l1 = np.round((np.random.rand(self.dim_states, self.dim_l1) - 0.5) * 12, 1)
        self.bias_l1 = np.round(np.random.rand(1, self.dim_l1) - 0.5 * 12, 1)
        self.weight = np.round((np.random.rand(self.dim_l1, self.dim_actions) - 0.5) * 12, 1)
        self.bias = np.round(np.random.rand(1, self.dim_actions) - 0.5 * 12, 1)

    def get_action(self, state):
        state = state.T

        linear_output_l1 = np.matmul(state, self.weight_l1) + self.bias_l1
        layer1_output = np.maximum(0.2 * linear_output_l1, linear_output_l1)
        return np.matmul(layer1_output, self.weight) + self.bias

    def __str__(self):
        output = "Weights L1:\n"
        for w in self.weight_l1:
            output += ", ".join([str(i) for i in w])
            output += "\n"
        output += "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        output += "Bias L1:\n"
        for b in self.bias_l1:
            output += ", ".join([str(i) for i in b])
            output += "\n"
        output += "Bias:\n"
        for b in self.bias:
            output += ", ".join([str(i) for i in b])
            output += "\n"

        return output

    def update_policy(self, weight_and_bias_list):
        if weight_and_bias_list is None:
            return
        
        self.weight_l1 = np.array(weight_and_bias_list[0])
        self.bias_l1 = np.array(weight_and_bias_list[1])
        self.weight = np.array(weight_and_bias_list[2])
        self.bias = np.array(weight_and_bias_list[3])

    def get_parameters(self, layer=None):
        if layer == 1:
            return np.concatenate((self.weight_l1.reshape(-1), self.bias_l1.reshape(-1)))
        elif layer == 2:
            return np.concatenate((self.weight.reshape(-1), self.bias.reshape(-1)))
        else:
            parameters = np.concatenate((self.weight_l1.reshape(-1), self.bias_l1.reshape(-1)))
            parameters = np.concatenate((parameters, self.weight.reshape(-1)))
            parameters = np.concatenate((parameters, self.bias.reshape(-1)))
        return parameters
