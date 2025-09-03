import time
import numpy as np
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain import LLMBrainStandardized
from world.pong import PongWorld


class PongAgent(ContinuousAgent):
    def __init__(
        self,
        num_episodes,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_ui_template,
        llm_output_conversion_template,
        llm_model_name,
        model_type,
        base_model,
        num_evaluation_episodes,
        step_size = 1.0,
        reset_llm_conversations = False,
        env_desc_file = None
    ):
        self.reset_llm_conversations = reset_llm_conversations
        self.max_val = 3.0
        self.step_size = step_size

        super().__init__(
            num_episodes, logdir, actions, states, max_traj_count, \
            max_traj_length, llm_si_template, llm_ui_template, llm_output_conversion_template, \
            llm_model_name, model_type, base_model, num_evaluation_episodes, env_desc_file
        )


    def get_next_action(self, state):
        action = self.policy.get_action(state).argmax()
        return action