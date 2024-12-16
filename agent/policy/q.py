from agent.policy.base_policy import Policy
import itertools
import random


# states = [
#   [0, 1, 2], # dim 1
#   [0, 1, 2, 3], # dim 2
#   ...
#   [0, 1, 2]  # dim n
# ]

# actions = [
#   [0, 1, 2], # dim 1
#   [0, 1, 2, 3], # dim 2
#   ...
#   [0, 1, 2]  # dim m
# ]
class QTable(Policy):
    def __init__(self, states, actions):
        super().__init__(states, actions)
        self.q_table_length = self._calculate_q_table_length(states, actions)
        print(f"Q Table length: {self.q_table_length}")
        if self.q_table_length > 1000:
            raise Exception("Q Table is too large to handle")
        
        self.initialize_policy()


    def _calculate_q_table_length(self, states, actions):
        length = 1
        for state in states:
            length *= len(state)
        for action in actions:
            length *= len(action)
        return length


    def initialize_policy(self):
        """
        Initializes the policy mapping for the agent.
        This method creates a nested dictionary structure where each state-action pair
        is assigned a random value. The states and actions are generated using the 
        Cartesian product of the provided states and actions lists.
        Attributes:
            self.mapping (dict): A dictionary where keys are states (tuples) and values 
                                 are dictionaries. The inner dictionaries have actions 
                                 (tuples) as keys and random float values as values.
        """

        self.mapping = dict()
        for state in itertools.product(*self.states):
            self.mapping[state] = dict()
            for action in itertools.product(*self.actions):
                self.mapping[state][action] = random.random()


    def get_action(self, state):
        """
        Returns the action with the highest Q-value for the given state.
        Args:
            state (tuple): The state for which to select an action.
        Returns:
            tuple: The action with the highest Q-value.
        """
        best_action = None
        best_value = float('-inf')
        for action, value in self.mapping[state].items():
            if value > best_value:
                best_value = value
                best_action = action
        return best_action



    def __str__(self):
        """
        Returns a string representation of the Q-table.
        Returns:
            str: A string representation of the Q-table.
        """
        # TODO: Change the title of the table to the name of each dim, e.g., "cos(theta) | sin(theta) | velocity | action | q_value"
        table = ["State | Action | Q-Value"]
        for state, actions in self.mapping.items():
            for action, value in actions.items():
                table.append(f"{state} | {action} | {value}")
        return "\n".join(table)


    def update_q_value(self, state, action, new_q_value):
        if state in self.mapping and action in self.mapping[state]:
            self.mapping[state][action] = new_q_value

    
    def update_policy(self, new_q_table):
        """
        Updates the policy with a new Q-table.
        This method iterates over the provided new Q-table and updates the Q-values
        for each state-action pair using the `update_q_value` method.
        Args:
            new_q_table (list of tuples): A list where each element is a tuple containing
                                          (state, action, q_value). `state` is the current state,
                                          `action` is the action taken, and `q_value` is the
                                          corresponding Q-value.
        Example:
            new_q_table = [
                (state1, action1, q_value1),
                (state2, action2, q_value2),
                ...
            ]
            policy.update_policy(new_q_table)
        """
        
        for state, action, q_value in new_q_table:
            self.update_q_value(state, action, q_value)
