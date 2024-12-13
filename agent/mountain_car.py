from agent.policy.q import QTable
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain import LLMBrain
from world.mountain_car import MountainCarWorld


class MountainCarAgent:
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
    ):
        self.q_table = QTable(actions=actions, states=states)
        self.replay_buffer = ReplayBuffer(
            max_traj_count=max_traj_count, max_traj_length=max_traj_length
        )
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes

    def rollout_episode(self, world: MountainCarWorld):
        state = world.reset()
        self.replay_buffer.start_new_trajectory()
        done = False
        while not done:
            action = self.q_table.get_action(state)
            next_state, reward, done = world.step(action)
            self.replay_buffer.add_step(state, action, reward)
            state = next_state
        return world.get_accu_reward()

    def train_policy(self, world: MountainCarWorld):
        # Run the episode and collect the trajectory
        print("Rolling out an episode...")
        result = self.rollout_episode(world)
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_q_values_list, reasoning = self.llm_brain.llm_update_q_table(
            self.q_table, self.replay_buffer
        )
        self.q_table.update_policy(new_q_values_list)
        print("Policy updated!")

    def evaluate_policy(self, world: MountainCarWorld):
        results = []
        for _ in range(self.num_evaluation_episodes):
            result = self.rollout_episode(world)
            results.append(result)
        return results
