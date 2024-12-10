from world.base_world import BaseWorld
import gym


class MountainCarWorld(BaseWorld):
    def __init__(self):
        super().__init__("MountainCar-v0")
