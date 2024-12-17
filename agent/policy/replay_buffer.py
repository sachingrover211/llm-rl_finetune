from collections import deque
import numpy as np
np.set_printoptions(precision=2)


class TrajectoryBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward):
        if self.count < self.buffer_size:
            self.buffer.append((state, action, reward))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((state, action, reward))

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def get_trajectory(self):
        return self.buffer

    def __str__(self):
        buffer_table = "State | Action | Reward\n"
        for state, action, reward in self.buffer:
            buffer_table += f"{state} | {action} | {reward}\n"
        return buffer_table


class ReplayBuffer:
    def __init__(self, max_traj_count, max_traj_length):
        self.buffer_size = max_traj_count
        self.max_traj_length = max_traj_length
        self.count = 0
        self.buffer = deque()
    
    def start_new_trajectory(self):
        if self.count >= self.buffer_size:
            self.buffer.popleft()
            self.count -= 1
        self.buffer.append(TrajectoryBuffer(self.max_traj_length))
        self.count += 1

    def add_step(self, state, action, reward):
        if len(self.buffer) == 0:
            raise Exception("Replay buffer is empty. Start a new trajectory first.")
        self.buffer[-1].add(state, action, reward)

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0
    
    def __str__(self):
        buffer_table = ""
        for i, trajectory in enumerate(self.buffer):
            buffer_table += f"Trajectory {i}:\n"
            buffer_table += str(trajectory)
            buffer_table += "\n"
        return buffer_table
