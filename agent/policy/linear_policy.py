import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)
        self.dim_states = dim_states
        self.dim_actions = dim_actions


    def initialize_policy(self):
        #self.weight = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        #self.bias = [[0.0, 0.0]]
        self.weight = np.random.uniform(-3.5, 3.5, (self.dim_states, self.dim_actions))
        self.bias = np.random.uniform(-3.5, 3.5, (1, self.dim_actions))
        self.weight = np.round(self.weight, decimals = 4)
        self.bias = np.round(self.bias, decimals = 4)


    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
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
        copy_weight = np.copy(self.weight)
        copy_bias = np.copy(self.bias)
        try:
            if weight_and_bias_list is None:
                return
            self.weight = np.array(weight_and_bias_list[:-1])
            self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)
        except Exception as e:
            print("Updating policy error", e)
            self.weight = copy_weight
            self.bias = copy_bias


class LinearContinuousActionPolicy(LinearPolicy):
    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)
        sigmoid = lambda x: 1 / (1+np.exp(-x))
        self.sigmoid_v = np.vectorize(sigmoid)


    def get_action(self, state):
        actions = super().get_action(state)
        return np.squeeze(self.sigmoid_v(actions))

