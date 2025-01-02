from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.halfcheetah_continuous import HalfcheetahContinuousWorld
import numpy as np


class HalfcheetahLinearAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
    ):
        self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = ReplayBuffer(
            max_traj_count=max_traj_count, max_traj_length=max_traj_length
        )
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

    def rollout_episode(self, world: HalfcheetahContinuousWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0).T
        if record:
            self.replay_buffer.start_new_trajectory()
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done = world.step(action)
            if record and step_idx % 10 == 0:
                self.replay_buffer.add_step(state.T[0], action[0], reward)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            state = next_state
            step_idx += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        return world.get_accu_reward()

    def train_policy(self, world: HalfcheetahContinuousWorld, logdir):

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logging_file)
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list, (reasoning, reasoning_processed) = (
            self.llm_brain.llm_update_parameters(
                self.policy, self.replay_buffer
            )
        )
        self.policy.update_policy(new_parameter_list)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        q_reasoning_processed_filename = f"{logdir}/parameters_reasoning_processed.txt"
        q_reasoning_processed_file = open(q_reasoning_processed_filename, "w")
        q_reasoning_processed_file.write(reasoning_processed)
        q_reasoning_processed_file.close()
        print("Policy updated!")

        self.training_episodes += 1

    def evaluate_policy(self, world: HalfcheetahContinuousWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
