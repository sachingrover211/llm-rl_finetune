import time
import numpy as np
from agent.policy.q import QTable
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain import LLMBrainStandardized
from world.cartpole import CartpoleWorld


class CartpoleAgent:
    def __init__(
        self,
        num_episodes,
        logdir,
        actions,
        states,
        num_theta_bins,
        num_podition_bins,
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
        self.replay_buffer = EpisodeRewardBufferNoBias(
            max_size=max_traj_count
        )

        self.llm_brain = LLMBrainStandardized(
            llm_si_template, llm_output_conversion_template, llm_model_name, env_desc_file, model_type, base_model, env_desc_file, num_episodes
        )
        self.llm_brain.reset_llm_conversation()

        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.replay_table_size = 100
        self.average_reward = 0
        self.max_val = 500.0
        self.step_size = step_size
        self.reset_llm_conversations = reset_llm_conversations
        self.record = True
        self.run_time = 0.0


    def initialize_policy(self, states, actions):
        self.policy = QTable(actions=actions, states=states)


    def rollout_episode(self, world, logdir, logging_file, record_video = False):
        state = world.reset()

        logging_file.write(f"state | action | reward\n")
        done = False
        while not done:
            action = self.policy.get_action(state).argmax()
            next_state, reward, done = world.step(action, record_video)
            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state

        return world.get_accu_reward()

    def train_policy(self, world, logdir):
        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logdir, logging_file, self.record_video)
        logging_file.close()
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_q_values_list, reasoning = self.llm_brain.llm_update_q_table(
            self.policy, self.replay_buffer
        )
        self.policy.update_policy(new_q_values_list)
        logging_q_filename = f"{logdir}/q_table.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        self.q_reasoning_filename = f"{logdir}/q_reasoning.txt"
        self.q_reasoning_file = open(self.q_reasoning_filename, "w")
        self.q_reasoning_file.write(reasoning)
        self.q_reasoning_file.close()
        print("Policy updated!")

        self.training_episodes += 1

    def evaluate_policy(self, world, logdir):
        results = []
        eval_start_time = time.time()

        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logdir, logging_file)
            results.append(result)
            logging_file.close()

        if self.record:
            self.replay_buffer.add(
                self.policy.get_parameters(), np.mean(results)
            )

        eval_time = time.time() - eval_start_time
        return results, eval_time


class ContinuousCartpoleAgent(CartpoleAgent):
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
        step_size,
        reset_llm_conversations,
        env_desc_file
    ):
        super().__init__(num_episodes, logdir, action_dims, state_dims, 0, 0, \
                    max_traj_count, max_traj_length, \
                    llm_si_template, llm_ui_template, llm_output_conversation_template, \
                    llm_model_name, model_type, base_model, num_evaluation_episodes, \
                    step_size, reset_llm_conversations, env_desc_file)


    def initialize_policy(self, state, actions):
        self.policy = LinearPolicy(state, actions)
        self.policy.initialize_policy()
        self.llm_brain.matrix_size = (state + 1, actions) # for the bias
        self.rank = (state + 1)*actions
        self.llm_brain.policy = self.policy


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
        self.llm_brain.policy = self.policy

        # if we just want to track current conversation only
        if self.reset_llm_conversations:
            self.llm_brain.reset_llm_conversation()

        updated_matrix, reasoning, self.run_time = self.llm_brain.llm_update_linear_policy(
            self.replay_buffer, self.training_episodes + 1, self.rank, self.max_val, self.step_size
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
        logging_matrix_file.write(str(self.policy) + "\n\nLLM runtime: "+ str(self.run_time))
        logging_matrix_file.close()

        # logging the response from the llm
        self.matrix_reasoning_filename = f"{logdir}/matrix_with_reasoning.txt"
        self.matrix_reasoning_file = open(self.matrix_reasoning_filename, "w")
        self.matrix_reasoning_file.write(reasoning)
        self.matrix_reasoning_file.close()
        print("Policy updated!")

        self.training_episodes += 1
        self.llm_brain.episode = self.training_episodes
