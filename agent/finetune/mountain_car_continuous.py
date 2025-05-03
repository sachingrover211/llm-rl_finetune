from agent.finetune.base_agent import BaseFinetuneAgent


class MountainCarFinetuneAgent(BaseFinetuneAgent):
    def __init__(
        self,
        actions,
        states,
        num_evaluation_episodes,
    ):
        super().__init__(actions, states, num_evaluation_episodes)
