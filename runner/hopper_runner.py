from world.hopper import HopperWorld
from agent.hopper import HopperAgent
from runner.training_loop import training_loop


def run_training_loop(**kwargs):
    # world_class & agent_class are passed here, everything else comes from YAML
    return training_loop(world_class=HopperWorld,
                     agent_class=HopperAgent,
                     **kwargs)