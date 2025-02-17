from agent.policy.q import QTable
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain import LLMBrain
from world.frozen_lake import FrozenLakeWorld

class FrozenLakeAgent:
    def __init__(
        self,
        logdir,
        actions,
        grid_size,
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
                max_traj_count = max_traj_count, max_traj_length = max_traj_length
            )

        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversation_template, llm_model_name, llm_ui_template
        )
        self.llm_brain.reset_llm_conversation()

        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.record_video = record_video
        self.replay_table_size = 100
        self.average_reward = 0
        self.use_replay_buffer = use_replay_buffer
        self.reset_llm_conversations = reset_llm_conversations


    def initialize_policy(self, grid_size, actions):
        state_dim = grid_size*grid_size
        self.policy = QTable(actions=actions, states=list(range(state_dim)))
        self.training_episodes = 0
        self.llm_brain.q_dim = (state_dim, len(actions[0]))


    def rollout_episode(self, world, logdir, logging_file):
        state = world.reset()

        if self.use_replay_buffer:
            self.replay_buffer.start_new_trajectory()

        logging_file.write("state | action | reward\n")
        done = False
        truncated = False

        while not (done or truncated):
            action = self.policy.get_Action(state)
            next_state, reward, done, truncated = world.step(action)
            if self.use_replay_buffer:
                self.repla_buffer.add_step(state, action, reward)

            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state

        world.close()
        return done, world.get_accu_reward()


    def train_policy(self, world, logdir):
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logdir, logging_file)
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
        q_reasoning_filename = f"{logdir}/q_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()

        print("Policy updated!")

        request = [req[self.llm_brain.TEXT_KEY] for req in self.llm_brain.llm_conversation]
        request = "\n#################\n".join(request)
        logging_request_filename = f"{logdir}/request.txt"
        with open(logging_request_filename, "w") as f:
            f.write(request)

        self.training_episodes += 1
        self.llm_brain.episode = self.training_episodes


    def evaluate_policy(self, world, logdir):
        results = []
        completed_instances = 0
        if self.use_replay_buffer:
            self.replay_buffer.clear()

        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            done, result = self.rollout_episode(world, logdir, logging_file)
            results.append(result)
            if done:
                completed_instances += 1

            logging_file.close()

        return results, completed_instances

