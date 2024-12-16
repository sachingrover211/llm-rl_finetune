import numpy as np
from agent.policy.base_polich import Policy


class LinearPolicy(Policy):
    def __init__(self, state, actions):
        super().__init__(state, actions)

        self.dim_states = len(states)
        self.dim_actions = len(actions)
        # I dont think we need this but saving none the less
        self.state = state
        self.actions = actions


    def initialize_policy(self):
        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)
        

    def get_action(self, state):
        return self.weight*state + self.bias


    def __str__(self)
        output = ""
        for w in self.weight:
            output = "\t".join([str(i) for i in w])
            output += "\n"

        output += "\t".join([str(i) for i in b])

        return output


    def update_policy(self, weight):
        self.weight = weight[:-1]
        self.bias = weight[-1]

