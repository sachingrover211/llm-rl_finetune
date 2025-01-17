from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.mountaincar_continuous_action import MountaincarContinuousActionWorld
import numpy as np
import re


class MountaincarContinuousActionLLMNumOptimAgent:
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
        self.replay_buffer = EpisodeRewardBuffer(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

    def norm_state(self, state):
        state_shape = state.shape
        state = state.reshape(-1)
        state = (state + np.array([0.3, 0.0])) * np.array([2.0, 1.0]) / np.array([1.8, 0.07])
        return state.reshape(state_shape)

    def sigmoid(self, z):
        return (1/(1 + np.exp(-z))) * 2 - 1
    
    def rollout_episode(self, world: MountaincarContinuousActionWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(f"{self.policy.weight.T[0][0]}, {self.policy.weight.T[0][1]}, {self.policy.bias[0][0]}\n")
        logging_file.write(f"parameter ends")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.sigmoid(self.policy.get_action(self.norm_state(state).T))
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            state = next_state
            step_idx += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        if record:
            self.replay_buffer.add(
                self.policy.weight, self.policy.bias, world.get_accu_reward()
            )
        return world.get_accu_reward()

    def random_warmup(self, world: MountaincarContinuousActionWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def train_policy(self, world: MountaincarContinuousActionWorld, logdir, search_std):

        def parse_parameters(input_text):
            # This regex looks for integers or floating-point numbers (including optional sign)
            s = input_text.split("\n")[0]
            pattern = r"[-+]?\d+(?:\.\d+)?"
            matches = re.findall(pattern, s)

            # Convert matched strings to float (or int if you prefer to differentiate)
            results = []
            for match in matches:
                results.append(float(match))
            assert len(results) == 3
            return np.array(results).reshape((3, 1))

        def str_3d_examples(replay_buffer: EpisodeRewardBuffer):

            all_parameters = []
            for weights, bias, reward in replay_buffer.buffer:
                parameters = np.concatenate((weights, bias))
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(3):
                    l += f'{"abcdefghijklmnopqr"[i]}: {parameters[i]}; '
                fxy = reward
                l += f"f(a,b,c): {fxy}\n"
                text += l
            return text

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logging_file)
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list, reasoning = self.llm_brain.llm_update_parameters_num_optim(
            str_3d_examples(self.replay_buffer),
            parse_parameters,
            self.training_episodes,
            search_std,
        )

        print(self.policy.weight.shape, self.policy.bias.shape)
        print(new_parameter_list.shape)
        self.policy.update_policy(new_parameter_list)
        print(self.policy.weight.shape, self.policy.bias.shape)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        self.training_episodes += 1

    def evaluate_policy(self, world: MountaincarContinuousActionWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
