from world.walker_2d import Walker2dWorld
from agent.walker_2d import Walker2dAgent
from runner.training_loop import training_loop


def run_training_loop(**kwargs):
    # world_class & agent_class are passed here, everything else comes from YAML
    return training_loop(world_class=Walker2dWorld,
                     agent_class=Walker2dAgent,
                     **kwargs)