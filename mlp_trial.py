import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# =====================
# Replay Buffer
# =====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# =====================
# MLP Gaussian Policy
# =====================
class MLPGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(MLPGaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # A simple MLP with two hidden layers
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.xavier_uniform_(self.log_std_head.weight)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs, deterministic=False):
        mean, log_std = self(obs)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        return action, mean, log_std

    def log_prob(self, obs, action):
        # Compute log probability of actions under this policy
        mean, log_std = self(obs)
        std = log_std.exp()

        # Convert action back from tanh-space to Gaussian parameter space
        # If a = tanh(x), then x = arctanh(a) = 0.5 * log((1+a)/(1-a))
        pre_tanh = 0.5 * torch.log((1 + action) / (1 - action + 1e-6) + 1e-6)

        var = std ** 2
        log_prob_gaussian = -0.5 * ((pre_tanh - mean) ** 2 / (var+1e-6) + 2 * log_std + np.log(2 * np.pi))
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1, keepdim=True)

        # Jacobian of the tanh transformation
        log_det_jacobian = torch.sum(torch.log(1 - action**2 + 1e-6), dim=-1, keepdim=True)

        log_prob = log_prob_gaussian - log_det_jacobian
        return log_prob

# =====================
# Q-Network (Critics)
# =====================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau*s_param.data + (1.0 - tau)*t_param.data)

def main():
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cpu")  # Change to "cuda" if using GPU

    # Hyperparameters
    gamma = 0.99
    tau = 0.005
    alpha_lr = 1e-3
    actor_lr = 3e-4
    critic_lr = 3e-4
    buffer_capacity = 100000
    batch_size = 256
    max_steps = 300000
    start_steps = 10000
    update_every = 50
    eval_every = 5000

    replay_buffer = ReplayBuffer(buffer_capacity)

    # Initialize networks
    actor = MLPGaussianPolicy(obs_dim, act_dim).to(device)
    q1 = QNetwork(obs_dim, act_dim).to(device)
    q2 = QNetwork(obs_dim, act_dim).to(device)
    q1_target = QNetwork(obs_dim, act_dim).to(device)
    q2_target = QNetwork(obs_dim, act_dim).to(device)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Target entropy heuristic: -dim(A)
    target_entropy = -act_dim

    # Entropy coefficient (alpha)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=alpha_lr)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    q1_optim = optim.Adam(q1.parameters(), lr=critic_lr)
    q2_optim = optim.Adam(q2.parameters(), lr=critic_lr)

    state, _ = env.reset(seed=42)
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    total_steps = 0

    for step in range(max_steps):
        total_steps += 1
        # Select action
        if step < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                a, _, _ = actor.sample(obs_tensor, deterministic=False)
                action = a.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store in buffer
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_steps += 1
        episode_reward += reward

        if done:
            state, _ = env.reset()
            episode_count += 1
            episode_steps = 0
            episode_reward = 0

        # Update networks
        if len(replay_buffer) > batch_size and step % update_every == 0:
            for _ in range(update_every):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)

                with torch.no_grad():
                    # Next action and log_prob
                    next_action, _, _ = actor.sample(next_states)
                    next_log_prob = actor.log_prob(next_states, next_action)
                    # Compute target Q
                    q1_next = q1_target(next_states, next_action)
                    q2_next = q2_target(next_states, next_action)
                    q_next = torch.min(q1_next, q2_next) - torch.exp(log_alpha)*next_log_prob
                    q_target = rewards + (1 - dones)*gamma*q_next

                # Update Q networks
                q1_val = q1(states, actions)
                q2_val = q2(states, actions)
                q1_loss = F.mse_loss(q1_val, q_target)
                q2_loss = F.mse_loss(q2_val, q_target)
                q_loss = q1_loss + q2_loss

                q1_optim.zero_grad()
                q2_optim.zero_grad()
                q_loss.backward()
                q1_optim.step()
                q2_optim.step()

                # Update Actor
                new_action, _, _ = actor.sample(states)
                log_prob = actor.log_prob(states, new_action)
                q1_new = q1(states, new_action)
                q2_new = q2(states, new_action)
                q_new_min = torch.min(q1_new, q2_new)

                actor_loss = (torch.exp(log_alpha)*log_prob - q_new_min).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # Update alpha (entropy temperature)
                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                # Soft update targets
                soft_update(q1_target, q1, tau)
                soft_update(q2_target, q2, tau)

        # Evaluate occasionally
        if step % eval_every == 0 and step > 0:
            eval_returns = []
            for _ in range(5):
                s, _ = env.reset()
                ep_ret = 0
                done_eval = False
                while not done_eval:
                    with torch.no_grad():
                        s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                        a, _, _ = actor.sample(s_tensor, deterministic=True)
                        a = a.cpu().numpy()[0]
                    s, r, term, trunc, _ = env.step(a)
                    ep_ret += r
                    done_eval = term or trunc
                eval_returns.append(ep_ret)
            print(f"Step: {step}, Average Eval Return: {np.mean(eval_returns):.2f}")

    env.close()

if __name__ == "__main__":
    main()

