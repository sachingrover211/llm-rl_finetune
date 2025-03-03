from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.linear_policy import LinearPolicy


class MountainCarFinetuneAgent:
    def __init__(
        self,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        finetune_model,
        dataset_size,
        use_replay_buffer,
        num_evaluation_episodes,
    ):
        self.logdir = logdir
        self.dataset_size = dataset_size
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.use_replay_buffer = use_replay_buffer
        self.replay_buffer = None
        if use_replay_buffer:
            self.replay_buffer = ReplayBuffer(
                max_traj_count=max_traj_count, max_traj_length=max_traj_length
            )

        self.actions = actions
        self.states = states
        self.policy = LinearPolicy(self.states, self.actions)


    def initialize_policy(self, weight):
        self.policy.initialize_policy_with_weights(weight)


    def rollout_episode(self, world, logdir, logging_file, record_video = False):
        if record_video:
            state = world.reset_with_video(logdir, f"episode_{self.training_episodes}")
            world.start_video_recorder()
        else:
            state = world.reset()

        if self.use_replay_buffer:
            self.replay_buffer.start_new_trajectory()

        logging_file.write(f"state | action | reward\n")
        done = False
        while not done:
            action = [self.policy.get_action(state)]
            next_state, reward, done = world.step(action)
            if self.use_replay_buffer:
                self.replay_buffer.add_step(state, action, reward)
            logging_file.write(f"{state} | {action} | {reward}\n")
            state = next_state

        if record_video:
            world.close_video_recorder()

        world.close()
        return world.get_accu_reward()


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

