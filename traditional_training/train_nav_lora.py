import gymnasium as gym
import numpy as np


def evaluate_policy(env, W, num_episodes=1, max_steps=1000):
    """
    Run the policy (linear, no bias) on 'num_episodes' rollouts.
    Return the average total reward.
    """
    total_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            # Action = W @ state
            # print(obs.shape, W.shape)
            W = W.reshape(5, 3)
            action = np.dot(obs, W)  # shape (3,)
            action = np.argmax(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def train_linear_policy_cem(
    env_name="gym_navigation:NavigationTrack-v0",
    seed=123,
    num_iterations=10,
    population_size=50,
    elite_frac=0.2,
    init_std=0.5,
    rollout_episodes=1,
    max_steps=1000,
):
    """
    Train a linear policy using a derivative-free Cross-Entropy Method.
    Policy shape is [state_dim x action_dim], no bias.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create the MuJoCo Hopper environment (Gymnasium)
    env = gym.make(env_name)

    # Observation and action dimensions
    obs_dim = env.observation_space.shape[0]  # Should be 11 for Hopper-v5
    act_dim = 3  # Should be 3 for Hopper-v5

    # Verify dimensions
    assert obs_dim == 5, f"Expected obs_dim=5, got {obs_dim}"
    assert act_dim == 3, f"Expected act_dim=3, got {act_dim}"

    # CEM hyperparameters
    elite_num = int(np.round(population_size * elite_frac))
    rank = 3

    # Initialize mean and std of parameter distribution
    # mu = np.zeros((obs_dim, act_dim), dtype=np.float32)
    # sigma = np.ones((obs_dim, act_dim), dtype=np.float32) * init_std


    mu = np.zeros((obs_dim + act_dim, rank), dtype=np.float32)
    sigma = np.ones((obs_dim + act_dim, rank), dtype=np.float32) * init_std

    # mu_matrix = np.dot(mu[:obs_dim].reshape(obs_dim, rank), mu[obs_dim: ].reshape(rank, act_dim))
    # sigma_matrix = np.dot(sigma[:obs_dim].reshape(obs_dim, rank), sigma[obs_dim: ].reshape(rank, act_dim))

    best_W = None
    best_score = -np.inf

    log_file = open("logs/cem_lora_nav_rank_3_take_3_log.txt", "w")
    for iteration in range(num_iterations):
        # Sample parameter matrices
        params_batch = []
        rewards = []

        for _ in range(population_size):
            # Sample W from N(mu, sigma^2 I)
            # print(mu.shape, (sigma * np.random.randn(obs_dim + act_dim, 1)).shape, sigma.shape)
            W_candidate_lora = mu + sigma * np.random.randn(obs_dim + act_dim, 1)
            W_candidate = np.dot(W_candidate_lora[:obs_dim].reshape(obs_dim, rank), W_candidate_lora[obs_dim:].reshape(rank, act_dim))
            
            # Evaluate candidate
            reward = evaluate_policy(
                env, W_candidate, num_episodes=rollout_episodes, max_steps=max_steps
            )
            params_batch.append(W_candidate_lora)
            rewards.append(reward)

        # Convert to numpy arrays
        rewards = np.array(rewards)
        params_batch = np.array(params_batch)

        # Track the best so far
        idx_best = np.argmax(rewards)
        if rewards[idx_best] > best_score:
            best_score = rewards[idx_best]
            best_W = params_batch[idx_best]

        # Select top "elite" performers
        elite_indices = rewards.argsort()[::-1][:elite_num]
        elite_params = params_batch[elite_indices]

        # Update distribution to match elites
        mu = np.mean(elite_params, axis=0)
        sigma = np.std(elite_params, axis=0) + 1e-8  # avoid zero std

        print(
            f"Iteration {iteration+1}/{num_iterations} | "
            f"Best Reward So Far: {best_score:.2f} | "
            f"Mean Reward This Iteration: {np.mean(rewards):.2f}"
        )

        log_file.write(f"Mean Reward: {(np.mean(rewards)):.2f}\n")

    env.close()
    log_file.close()
    return best_W, best_score


def train_hopper_cem():
    # Hyperparameters (adjust to taste)
    ENV_NAME = "gym_navigation:NavigationTrack-v0"
    SEED = 123
    NUM_ITERATIONS = 400
    POP_SIZE = 50
    ELITE_FRAC = 0.2
    INIT_STD = 0.5
    ROLLOUT_EPISODES = 1
    MAX_STEPS = 1000

    # Train
    best_matrix, best_return = train_linear_policy_cem(
        env_name=ENV_NAME,
        seed=SEED,
        num_iterations=NUM_ITERATIONS,
        population_size=POP_SIZE,
        elite_frac=ELITE_FRAC,
        init_std=INIT_STD,
        rollout_episodes=ROLLOUT_EPISODES,
        max_steps=MAX_STEPS,
    )

    print("\n========== Training Finished ==========")
    print("Best Weight Matrix (W) found:\n", best_matrix)
    print(f"Best observed return: {best_return:.2f}")



if __name__ == "__main__":
    train_hopper_cem()
    # train_hopper_cem_bias()