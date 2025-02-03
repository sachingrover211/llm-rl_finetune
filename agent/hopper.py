from agent.policy.linear_policy import LinearContinuousActionPolicy
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain import LLMBrain
from world.hopper import HopperWorld


class HopperAgent:
    def __init__(
        self,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_ui_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        record_video = False,
        use_replay_buffer = True,
        reset_llm_conversations = False,
    ):
        self.replay_buffer = None
        if use_replay_buffer:
            self.replay_buffer = ReplayBuffer(
                max_traj_count=max_traj_count, max_traj_length=max_traj_length
            )

        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name, llm_ui_template
        )
        self.llm_brain.reset_llm_conversation()

        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.record_video = record_video
        self.replay_table_size = max_traj_length
        self.average_reward = 0
        self.use_replay_buffer = use_replay_buffer
        self.reset_llm_conversations = reset_llm_conversations


    def initialize_policy(self, states, actions):
        self.policy = LinearContinuousActionPolicy(dim_actions=actions, dim_states=states)
        self.policy.initialize_policy()
        self.llm_brain.matrix_size = (states + 1, actions) # +1 for the bias
        self.llm_brain.create_regex()


    def rollout_episode(self, world, logdir, logging_file, record_video = False):
        if record_video:
            world.render_mode = "rgb_array"

        state = world.reset()

        if self.use_replay_buffer:
            self.replay_buffer.start_new_trajectory()

        logging_file.write(f"state | action | reward\n")
        done = False
        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done = world.step(action)
            if self.use_replay_buffer:
                self.replay_buffer.add_step(state, action, reward)
            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state
        
        return world.get_accu_reward()


    def train_policy(self, world, logdir = ""):
        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logdir, logging_file)
        logging_file.close()
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")

        replay_buffer_string = None
        if self.use_replay_buffer:
            index, samples = self.replay_buffer.sample_contiguous(self.replay_table_size)
            replay_buffer_string = self.replay_buffer.print_trajectory(index, samples)

        # if we just want to track current conversation only
        if self.reset_llm_conversations:
            self.llm_brain.reset_llm_conversation()

        if self.use_replay_buffer:
            updated_matrix, reasoning = self.llm_brain.llm_update_linear_policy(
                self.policy, self.average_reward, replay_buffer_string
            )
        else:
            updated_matrix, reasoning = self.llm_brain.llm_update_linear_policy(
                self.policy, self.average_reward, None
            )

        # logging request
        request = [req[self.llm_brain.TEXT_KEY] for req in self.llm_brain.llm_conversation]
        request = "\n#################\n".join(request)
        logging_request_filename = f"{logdir}/request.txt"
        with open(logging_request_filename, "w") as f:
            f.write(request)

        # logging updated_matrix value
        self.policy.update_policy(updated_matrix)
        logging_matrix_filename = f"{logdir}/matrix.txt"
        logging_matrix_file = open(logging_matrix_filename, "w")
        logging_matrix_file.write(str(self.policy))
        logging_matrix_file.close()

        # logging the response from the llm
        self.matrix_reasoning_filename = f"{logdir}/matrix_with_reasoning.txt"
        self.matrix_reasoning_file = open(self.matrix_reasoning_filename, "w")
        self.matrix_reasoning_file.write(reasoning)
        self.matrix_reasoning_file.close()
        print("Policy updated!")

        self.training_episodes += 1
        self.llm_brain.episode = self.training_episodes


    def evaluate_policy(self, world, logdir):
        results = []
        if self.use_replay_buffer:
            self.replay_buffer.clear()

        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logdir, logging_file)
            results.append(result)
            logging_file.close()
        return results


