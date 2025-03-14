from agent.policy.q import QTable
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re


class LLMNumOptimQTableRndmPrjAgent:
    def __init__(
        self,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        num_training_rollouts,
        rank,
        search_std,
        optimum,
    ):
        self.actions = actions
        self.states = states
        self.optimum = optimum
        self.search_std = search_std


        self.q_table = QTable(actions=actions, states=states)
        self.num_params = self.q_table.q_table_length

        self.G = np.random.randn(self.num_params, self.num_params)
        self.Q, self.R = np.linalg.qr(self.G)
        self.high_to_low_projection_matrix = self.Q[:, :rank]
        self.low_to_high_projection_matrix = self.Q[:, :rank].T
        self.rank = rank
        
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
    
    def parameters_high_to_low(self, parameters):
        return parameters.reshape(-1) @ self.high_to_low_projection_matrix
    
    def parameters_low_to_high(self, parameters):
        return parameters.reshape(-1) @ self.low_to_high_projection_matrix

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.q_table.get_action(state)
            action = int(np.reshape(action, (1,)))
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state
            step_idx += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        if record:
            self.replay_buffer.add(
                np.array(
                    self.q_table.get_policy_vector()
                ),
                world.get_accu_reward(),
            )
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.q_table.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            # This regex looks for integers or floating-point numbers (including optional sign)
            s = input_text.split("\n")[0]
            print('response:', s)
            pattern = re.compile(
                r'params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)'
            )
            matches = pattern.findall(s)

            # Convert matched strings to float (or int if you prefer to differentiate)
            results = []
            for match in matches:
                results.append(float(match[1]))
            print(results)
            assert len(results) == self.rank
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):

            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(n):
                    l += f'params[{i}]: {parameters[i]:.5g}; '
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text


        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list, reasoning = self.llm_brain.llm_update_parameters_num_optim(
            str_nd_examples(self.replay_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.search_std,
            self.rank,
            self.optimum,
        )

        print(new_parameter_list.shape)
        self.q_table.update_policy_vector(self.parameters_low_to_high(new_parameter_list))
        print(len(self.q_table.mapping))
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.q_table))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")


        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(20):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=False)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)

        self.training_episodes += 1

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
