import time
import numpy as np
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain import LLMBrainStandardized
from world.swimmer import SwimmerWorld


class SwimmerAgent:
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
        self.replay_buffer = EpisodeRewardBufferNoBias(
            max_size=num_episodes
        )

        self.llm_brain = LLMBrainStandardized(
            llm_si_template, llm_output_conversion_template, llm_model_name, env_desc_file, model_type, base_model, env_desc_file, num_episodes
        )
        self.llm_brain.reset_llm_conversation()

        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.replay_table_size = max_traj_length
        self.average_reward = 0
        self.reset_llm_conversations = reset_llm_conversations
        self.max_val = 250.0
        self.step_size = step_size
        self.record = True
        self.run_time = 0.0


    def initialize_policy(self, states, actions):
        self.policy = LinearPolicy(dim_actions=actions, dim_states=states)
        self.policy.initialize_policy()
        self.llm_brain.matrix_size = (states + 1, actions) # +1 for the bias
        self.llm_brain.create_regex()
        self.rank = (states + 1)*actions
        self.llm_brain.policy = self.policy


    def rollout_episode(self, world, logdir, logging_file):
        state = world.reset()
        #self.replay_buffer.start_new_trajectory()

        logging_file.write(f"state | action | reward\n")
        done = False
        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done = world.step(action)
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
        self.llm_brain.policy = self.policy

        # if we just want to track current conversation only
        if self.reset_llm_conversations:
            self.llm_brain.reset_llm_conversation()

        updated_matrix, reasoning, self.run_time = self.llm_brain.llm_update_linear_policy(
            self.replay_buffer, self.training_episodes +1, self.rank, self.max_val, self.step_size
        )
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


