from typing import List
import random

class Agent():
    """
        Base Agent is a random agent.
        Samples from the list of actions
    """
    def __init__(self, actions: List[str]):
        self.actions = actions
        self.name = "agent"


    def next(self, grid):
        return random.choice(self.actions)


