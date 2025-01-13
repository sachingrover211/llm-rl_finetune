from collections import deque
import numpy as np
import os
import random
np.set_printoptions(precision=2)


class TrajectoryBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        # tracks the number of steps for the trajectory
        self.step = 0
        # for ease of sampling changing deque to list ~ Sachin
        #self.buffer = deque()
        self.buffer = list()


    def add(self, state, action, reward):
        # instead of limiting the buffer size
        # changing it to randomly sample the information
        # Sachin -- my understanding is providing only the last part of the trace
        # can include unintended bias, uniformly sampling would make more sense.
        #if self.count < self.buffer_size:
        #    self.buffer.append((self.step, state, action, reward))
        self.count += 1
        #else:
        #    self.buffer.pop(0)
        #    self.buffer.popleft()
        self.buffer.append((self.step, state, action, reward))
        self.step += 1


    def size(self):
        return self.step


    def clear(self):
        # keeping count so that other code should not break
        self.buffer = list()
        self.step = 0
        self.count = 0


    def get_trajectory(self):
        return self.buffer


    def print_row(self, index):
        step, state, action, reward = self.buffer[index]
        return f"{step} | {state} | {action} | {reward}"


    def __str__(self):
        buffer_table = "Step | State | Action | Reward\n"
        for step, state, action, reward in self.buffer:
            buffer_table += f"{step} | {state} | {action} | {reward}\n"
        return buffer_table


class ReplayBuffer:
    def __init__(self, max_traj_count, max_traj_length):
        self.buffer_size = max_traj_count
        self.max_traj_length = max_traj_length
        self.count = 0
        # for ease of sampling
        # changing replay buffer from deque to list ~ Sachin
        self.buffer = list()
        #self.buffer = deque()
        self.total_size = 0


    def start_new_trajectory(self):
        #if self.count >= self.buffer_size:
        #    self.buffer.popleft()
        #    self.count -= 1
        self.buffer.append(TrajectoryBuffer(self.max_traj_length))
        self.count += 1


    def add_step(self, state, action, reward):
        if len(self.buffer) == 0:
            raise Exception("Replay buffer is empty. Start a new trajectory first.")
        self.buffer[-1].add(state, action, reward)
        self.total_size += 1


    def size(self):
        return self.count


    def clear(self):
        self.buffer.clear()
        self.count = 0


    def sample(self, number_of_rows):
        if self.total_size  < number_of_rows:
            return self.print_buffer()

        total_samples = [list(range(self.buffer[i].step)) for i in range(self.count)]
        samples = [[None] for _ in range(self.count)]
        for _ in range(number_of_rows):
            while True:
                traj_index = random.randint(0, self.count - 1)
                trajectory = total_samples[traj_index]
                if len(total_samples[traj_index]) == 0:
                    continue

                row_index = random.sample(total_samples[traj_index], 1)[0]
                #samp = self.buffer[traj_index].buffer[row_index]
                if not samples[traj_index][0]:
                    samples[traj_index][0] = row_index
                else:
                    samples[traj_index].append(row_index)

                total_samples[traj_index].remove(row_index)
                break

        for i in range(len(samples)):
            if samples[i][0] == None:
                samples[i][0] = random.sample(total_samples[i], 1)[0]
            samples[i].sort()

        return samples

    def sample_contiguous(self, number_of_rows):
        traj_index = random.randint(0, self.count - 1)
        trajectory = self.buffer[traj_index]
        samples = list()
        if len(trajectory.buffer) <= (number_of_rows + 1):
            return traj_index, list(range(0, len(trajectory.buffer)))

        start_index = random.randint(0, len(trajectory.buffer) - number_of_rows - 1)
        return traj_index, list(range(start_index, start_index + number_of_rows))


    def print_trajectory(self, traj_index, rows):
        output = list()
        output.append("Trajectory | Step | State | Action | Reward")
        trajectory = self.buffer[traj_index]
        for r_index in rows:
            output.append(f"{traj_index} | {trajectory.print_row(r_index)}")

        return "\n".join(output)


    def print_buffer(self):
        # this prints the buffer as one big table
        # with trajectory as part of the input
        output = list()
        output.append("Trajectory | Step | State | Action | Reward")
        for t in range(self.count):
            trajectory = self.buffer[t]
            for i in range(trajectory.step):
                output.append(f"{t} | {trajectory.print_row(i)}")

        return "\n".join(output)


    def print_samples(self, samples):
        output = list()
        output.append("Trajectory | Step | State | Action | Reward")

        for t in range(len(samples)):
            t_indices = samples[t]
            trajectory = self.buffer[t]
            for si in t_indices:
                output.append(f"{t} | {trajectory.print_row(si)}")

        return "\n".join(output)


    def __str__(self):
        buffer_table = ""
        for i, trajectory in enumerate(self.buffer):
            buffer_table += f"Trajectory {i}:\n"
            buffer_table += str(trajectory)
            buffer_table += "\n"
        return buffer_table


class EpisodeRewardBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, weights: np.ndarray, bias, reward):
        self.buffer.append((weights, bias, reward))

    def __str__(self):
        buffer_table = "Parameters | Reward\n"
        for weights, bias, reward in self.buffer:
            parameters = np.concatenate((weights, bias))
            buffer_table += f"{parameters.reshape(1, -1)} | {reward}\n"
        return buffer_table

    def load(self, folder):
        # Find all episode files
        all_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith('warmup_rollout')]
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(',')])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                self.add(parameters[:-1], parameters[-1:], rewards_mean)
                f.close()
        print(self)


class EpisodeRewardMeanStdBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, weights: np.ndarray, bias, reward_mean, reward_std):
        self.buffer.append((weights, bias, reward_mean, reward_std))

    def __str__(self):
        buffer_table = "Parameters | Reward_Mean | Reward_Std \n"
        for weights, bias, reward_mean, reward_std in self.buffer:
            parameters = np.concatenate((weights, bias))
            parameters = parameters.reshape(1, -1)[0]
            parameters = ', '.join([f"{x:.3f}" for x in parameters])
            buffer_table += f"{parameters} | {reward_mean} | {reward_std}\n"
        return buffer_table

    def load(self, folder):
        # Find all episode files
        all_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith('warmup_rollout')]
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(',')])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                rewards_std = np.std(rewards)
                self.add(parameters[:-1], parameters[-1:], rewards_mean, rewards_std)
                f.close()
        print(self)
