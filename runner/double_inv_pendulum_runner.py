from world.dbl_inv_pendulum import DIPWorld
from agent.dbl_inv_pendulum import DIPAgent
from runner.training_loop import training_loop


def run_training_loop(**kwargs):
    # world_class & agent_class are passed here, everything else comes from YAML
    return training_loop(world_class=DIPWorld,
                     agent_class=DIPAgent,
                     **kwargs)
