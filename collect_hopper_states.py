import numpy as np
import gymnasium as gym

def run_hopper_episodes(num_episodes=100):
    env = gym.make('Hopper-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    all_states = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        # Randomly initialize linear policy parameters
        policy_weights = (np.random.randn(action_dim, state_dim) - 0.5) * 6

        while not done:
            action = np.dot(policy_weights, state)
            state, reward, done, _, _ = env.step(action)
            all_states.append(state)

    env.close()
    return all_states

if __name__ == "__main__":
    states = run_hopper_episodes()
    states = np.array(states)
    print(states.shape)
    mean_states = np.mean(states, axis=(0))
    std_states = np.std(states, axis=(0))

    print("Mean of states:", mean_states)
    print("Standard deviation of states:", std_states)