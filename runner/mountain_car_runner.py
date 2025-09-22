from world.mountain_car import MountainCarWorld
from agent.mountain_car import MountainCarDiscreteActionAgent
from runner.training_loop import training_loop


def run_training_loop(**kwargs):
    # world_class & agent_class are passed here, everything else comes from YAML
    return training_loop(world_class=MountainCarWorld,
                     agent_class=MountainCarDiscreteActionAgent,
                     **kwargs)