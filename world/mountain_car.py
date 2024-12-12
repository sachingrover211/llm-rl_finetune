from world.base_world import BaseWorld
import gymnasium as gym


class MountainCarWorld(BaseWorld):
    def __init__(self):
        super().__init__("MountainCar-v0")

    def run(self):
        raise NotImplementedError("run method not implemented")