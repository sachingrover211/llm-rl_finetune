import re
import numpy as np
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
        self.matrix_size = (3, 1)


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

    def rollout_episode_without_logging(self, world):
        state = world.reset()
        done = False
        while not done:
            action = [self.policy.get_action(state)]
            next_state, reward, done = world.step(action)
            state = next_state

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


    def parse_response(self, response):
        new_parameters_list = list()
        float_continuous = "[0-9.-]+[.,\s\]\)\}|]+"
        single_float = "[0-9.-]+"
        new_ps = response.lower()

        if "<policy>" in new_ps:
            new_ps = self.get_sub_string(new_ps, "<policy>", "</policy>")

        sub_split = ""
        if "weights" in new_ps:
            sub_split = "weights"
        elif "weight matrix" in new_ps:
            sub_split = "weight matrix"

        if "- " in new_ps:
            new_ps = new_ps.replace("- ", "-")

        check_string = ["", new_ps]
        if sub_split != "":
            # the import string is right after the split
            check_string = new_ps.split(sub_split)

        param_size = self.matrix_size[0] * self.matrix_size[1]
        for i in range(1, len(check_string)):
            chk = check_string[i]
            temp = re.findall(float_continuous, chk)
            for t in temp:
                t = re.findall(single_float, t)[0]
                t = t.strip().strip(".").strip()
                if any(ch.isdigit() for ch in t):
                    try:
                        new_parameters_list.append(float(t))
                    except:
                        print(f"Error converting string {t} to float")
                        return list()

                if len(new_parameters_list) == param_size:
                    break

            if len(new_parameters_list) == param_size:
                break

        if len(new_parameters_list) == self.matrix_size[0]:
            return np.array(new_parameters_list).reshape(self.matrix_size)

        return list()


    def get_sub_string(self, response, start_tag, end_tag):
        start_index = response.find(start_tag)
        if start_index == -1:
            return ""

        end_index = response.find(end_tag)
        if end_index == -1:
            return ""

        start_index += len(start_tag)
        return response[start_index: end_index]
