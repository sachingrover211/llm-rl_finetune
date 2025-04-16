from agent.finetune.base_agent import BaseFinetuneAgent


class PongFinetuneAgent(BaseFinetuneAgent):
    def __init__(
        self,
        actions,
        states,
        num_evaluation_episodes,
    ):
        super().__init__(actions, states, num_evaluations_episodes)
