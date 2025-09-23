from agent.base_agent import ContinuousAgent

class HopperAgent(ContinuousAgent):
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
        warmup_episodes = 1,
        step_size = 1.0,
        reset_llm_conversations = False,
        env_desc_file = None
    ):
        self.reset_llm_conversations = reset_llm_conversations
        self.max_val = 1000.0
        self.step_size = step_size
        self.warmup_episodes = warmup_episodes
        
        super().__init__(
            num_episodes, logdir, actions, states, max_traj_count, \
            max_traj_length, llm_si_template, llm_ui_template, llm_output_conversion_template, \
            llm_model_name, model_type, base_model, num_evaluation_episodes, env_desc_file
        )