from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardMeanStdBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.swimmer_continuous import SwimmerContinuousWorld
import numpy as np


class SwimmerRandomSearchAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        num_evaluation_episodes,
    ):
        self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardMeanStdBuffer(max_size=max_traj_count)
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

    def rollout_episode(
        self, world: SwimmerContinuousWorld, logging_file, record=True, rollout_count=20
    ):
        accu_rewards = []
        for rollout_idx in range(rollout_count):
            state = world.reset()
            logging_file.write(str(self.policy))
            logging_file.write("parameter ends\n\n")
            logging_file.write(f"Rollout {rollout_idx}: \n")
            logging_file.write(f"state | action | reward\n")
            done = False
            step_idx = 0
            while not done:
                action = self.policy.get_action(state.T)
                next_state, reward, done = world.step(action)
                logging_file.write(f"{state.T} | {action[0]} | {reward}\n")
                state = next_state
                step_idx += 1
            logging_file.write(f"Total reward: {world.get_accu_reward()}\n\n")
            accu_rewards.append(world.get_accu_reward())
        accu_rewards = np.array(accu_rewards)
        mean_reward = np.mean(accu_rewards)
        std_reward = np.std(accu_rewards)
        if record:
            self.replay_buffer.add(self.policy.weight, self.policy.bias, mean_reward, std_reward)
        return mean_reward, std_reward

    def random_warmup(self, world: SwimmerContinuousWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")


    def random_search(self, replay_buffer: EpisodeRewardMeanStdBuffer, std=0.1):
        # Find the best parameters from the replay buffer
        best_parameters = None
        best_reward = -np.inf
        for weights, bias, reward_mean, reward_std in replay_buffer.buffer:
            parameters = np.concatenate((weights, bias))
            if reward_mean > best_reward:
                best_reward = reward_mean
                best_parameters = parameters
        
        # Generate new parameters
        new_parameters = np.random.normal(best_parameters, std)
        return new_parameters


    def train_policy(self, world: SwimmerContinuousWorld, logdir, std):

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        mean_reward, std_reward = self.rollout_episode(world, logging_file)
        print(f"Mean Reward: {mean_reward:.3f} Std Reward: {std_reward:.3f}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list = self.random_search(self.replay_buffer, std)

        print(self.policy.weight.shape, self.policy.bias.shape)
        print(new_parameter_list.shape)
        self.policy.update_policy(new_parameter_list)
        print(self.policy.weight.shape, self.policy.bias.shape)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        print("Policy updated!")

        self.training_episodes += 1

    def evaluate_policy(self, world: SwimmerContinuousWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False, rollout_count=1)
            results.append(result)
        return results
