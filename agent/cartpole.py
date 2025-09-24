import time
import numpy as np
from agent.base_agent import DiscreteAgent, ContinuousAgent

'''
' Not editing the QTable based version of the code.
' We moved to linear policy as most of the QTable based versions didnt work.
'''
class CartpoleAgent(DiscreteAgent):
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
        warmup_episodes=1,
        step_size = 1.0,
        reset_llm_conversations = False,
        env_desc_file = None
    ):

        self.reset_llm_conversations = reset_llm_conversations
        self.max_val = 500.0
        self.step_size = step_size
        self.warmup_episodes = warmup_episodes

        super().__init__(
            num_episodes, logdir, actions, states, max_traj_count, \
            max_traj_length, llm_si_template, llm_ui_template, llm_output_conversion_template, \
            llm_model_name, model_type, base_model, num_evaluation_episodes, env_desc_file
        )


    def get_next_action(self, state):
        action = self.policy.get_action(state).argmax()
        return action



class ContinuousCartpoleAgent(ContinuousAgent):
    def __init__(
        self,
        num_episodes,
        logdir,
        action_dims,
        state_dims,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_ui_template,
        llm_output_conversation_template,
        llm_model_name,
        model_type,
        base_model,
        num_evaluation_episodes,
        warmup_episodes,
        step_size,
        reset_llm_conversations,
        env_desc_file
    ):
        self.reset_llm_conversations = reset_llm_conversations
        self.max_val = 500.0
        self.step_size = step_size
        self.warmup_episodes = warmup_episodes

        super().__init__(
            num_episodes, logdir, action_dims, state_dims, max_traj_count, max_traj_length, \
            llm_si_template, llm_ui_template, llm_output_conversation_template, \
            llm_model_name, model_type, base_model, num_evaluation_episodes, env_desc_file
        )


    def get_next_action(self, state):
        action = self.policy.get_action(state).argmax()
        return action
