from agent.policy.base_policy import Policy

class QTable(Policy):
    def __init__(self, states, actions):
        super().__init__(states, actions)
        self.mapping = dict() # here it should be state and actions with q values or the state with best rated action adn q-value


    def get_action(self, state):
        pass



    def __str__(self):
        pass



    def initialize_policy(self):
        """
        can be randomized initialization of the mapping dictionary
        """
        pass



    def update_policy(self):
        """
        should be the algorithm for the updating it
        """
