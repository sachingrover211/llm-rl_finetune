from agent.policy.q_table import QTable
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain import LLMBrain
from world.frozen_lake import FrozenLakeWorld

class FrozenLakeAgent:
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
            warmup_episodes=1,
            step_size=1.0,
            reset_llm_conversations=False,
            env_desc_file=None,
            record_video=False,
            use_replay_buffer=True,
    ):
        self.replay_buffer = None
        if use_replay_buffer:
            self.replay_buffer = ReplayBuffer(
                max_traj_count=max_traj_count, max_traj_length=max_traj_length
            )

        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name,
            llm_ui_template, model_type, base_model
        )
        self.llm_brain.reset_llm_conversation()

        self.step_size = step_size
        self.warmup_episodes = warmup_episodes
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.record_video = record_video
        self.replay_table_size = 100
        self.average_reward = 0
        self.use_replay_buffer = use_replay_buffer
        self.reset_llm_conversations = reset_llm_conversations

        # Store states and actions for later use
        self.states = states
        self.actions = actions


    def initialize_policy(self, world, grid_size, actions):
        state_dim = grid_size*grid_size
        temp = list(range(state_dim))
        states = list()
        for state in temp:
            states.append(world.decode_state(state))
        self.policy = QTable(actions=actions, states=[states])
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
            action = self.policy.get_action(state)
            next_state, reward, done, truncated = world.step(action)
            if self.use_replay_buffer:
                self.replay_buffer.add_step(state, action, reward)

            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state

        world.close()
        return done, world.get_accu_reward(), world.get_total_steps()


    def train_policy(self, world, logdir, cost, completion_count):
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        result = self.rollout_episode(world, logdir, logging_file)
        logging_file.close()
        print(f"Result: {result}")

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        params = dict()
        params["map"] = "\n".join(world.grid)
        params["cost"] = cost
        params["count"] = completion_count

        replay_buffer_string = None
        if self.use_replay_buffer:
            index, samples = self.replay_buffer.sample_contiguous(self.replay_table_size)
            replay_buffer_string = self.replay_buffer.print_trajectory(index, samples)

        new_q_values_list, reasoning = self.llm_brain.llm_update_q_table(
            self.policy, replay_buffer_string, params
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
            done, reward, cost = self.rollout_episode(world, logdir, logging_file)
            results.append(cost)
            if done and reward > 0:
                completed_instances += 1

            logging_file.close()

        return results, completed_instances

