import gymnasium as gym
import numpy as np


OBS_DIM = None
ACT_DIM = None


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
            W = W.reshape(OBS_DIM, ACT_DIM)
            action = np.dot(obs, W)  # shape (3,)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def train_linear_policy_cem(
    env_name="Hopper-v5",
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
    act_dim = env.action_space.shape[0]  # Should be 3 for Hopper-v5

    # Verify dimensions
    assert obs_dim == 11, f"Expected obs_dim=11, got {obs_dim}"
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

    log_file = open("logs/cem_lora_hopper_rank_3_take_3_log.txt", "w")
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
    ENV_NAME = "Hopper-v5"
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


def evaluate_policy_bias(env, W, num_episodes=1, max_steps=1000):
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
            action = np.dot(obs, W[:-1]) + W[-1]  # shape (3,)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def train_linear_policy_cem_bias(
    env_name="Hopper-v5",
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
    act_dim = env.action_space.shape[0]  # Should be 3 for Hopper-v5

    # Verify dimensions
    assert obs_dim == 11, f"Expected obs_dim=11, got {obs_dim}"
    assert act_dim == 3, f"Expected act_dim=3, got {act_dim}"

    # CEM hyperparameters
    elite_num = int(np.round(population_size * elite_frac))

    # Initialize mean and std of parameter distribution
    mu = np.zeros((obs_dim + 1, act_dim), dtype=np.float32)
    sigma = np.ones((obs_dim + 1, act_dim), dtype=np.float32) * init_std

    best_W = None
    best_score = -np.inf

    for iteration in range(num_iterations):
        # Sample parameter matrices
        params_batch = []
        rewards = []

        for _ in range(population_size):
            # Sample W from N(mu, sigma^2 I)
            W_candidate = mu + sigma * np.random.randn(obs_dim + 1, act_dim)
            # Evaluate candidate
            reward = evaluate_policy_bias(
                env, W_candidate, num_episodes=rollout_episodes, max_steps=max_steps
            )
            params_batch.append(W_candidate)
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

    env.close()
    return best_W, best_score


def train_hopper_cem_bias():
    # Hyperparameters (adjust to taste)
    ENV_NAME = "Hopper-v5"
    SEED = 123
    NUM_ITERATIONS = 30
    POP_SIZE = 50
    ELITE_FRAC = 0.2
    INIT_STD = 5.0
    ROLLOUT_EPISODES = 1
    MAX_STEPS = 1000

    # Train
    best_matrix, best_return = train_linear_policy_cem_bias(
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



import cma  # pip install cma

# def evaluate_policy(env, params, num_episodes=1, max_steps=1000):
#     """
#     Evaluate a flattened parameter vector (length 33), 
#     reshape it into (11, 3) for the linear policy, 
#     and run it in Hopper-v5 for 'num_episodes' episodes.
    
#     Returns the average total reward.
#     """
#     # Reshape flat params -> W matrix
#     W = params.reshape(11, 3)
    
#     total_rewards = []
#     for _ in range(num_episodes):
#         obs, info = env.reset()
#         episode_reward = 0.0
#         for _ in range(max_steps):
#             action = np.dot(obs, W)  # shape (3,)
#             obs, reward, terminated, truncated, info = env.step(action)
#             episode_reward += reward
#             if terminated or truncated:
#                 break
#         total_rewards.append(episode_reward)
#     return np.mean(total_rewards)


def train_linear_policy_cma_es(env_name="Hopper-v5",
                               seed=123,
                               num_iterations=10,
                               population_size=16,
                               init_sigma=0.5,
                               rollout_episodes=1,
                               max_steps=1000,
                               exp_id=0,
                               rank=5):
    """
    Use CMA-ES to find a good linear policy W of shape (11, 3) 
    for the Hopper-v5 environment. Flattened dimension = 33.
    
    :param env_name: Name of the gymnasium environment.
    :param seed: Random seed.
    :param num_iterations: Maximum number of CMA-ES iterations (generations).
    :param population_size: Number of candidate solutions per generation 
                            (a.k.a. CMA-ES "lambda").
    :param init_sigma: Initial std. dev. for CMA-ES sampling.
    :param rollout_episodes: Number of episodes to average over when evaluating.
    :param max_steps: Max timesteps per episode.
    """
    np.random.seed(seed)
    
    # Create Hopper-v5 environment
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]  # Expect 11 for Hopper-v5
    act_dim = env.action_space.shape[0]       # Expect 3 for Hopper-v5
    global OBS_DIM, ACT_DIM
    OBS_DIM = obs_dim
    ACT_DIM = act_dim
    # assert obs_dim == 8, f"Expected obs_dim=8, got {obs_dim}"
    # assert act_dim == 2,  f"Expected act_dim=2, got {act_dim}"
    
    # Flatten dimension for linear W
    param_dim = obs_dim * act_dim  # 11*3=33
    G = np.random.randn(param_dim, param_dim)
    Q, R = np.linalg.qr(G)
    high_to_low_projection_matrix = Q[:, :rank]
    low_to_high_projection_matrix = Q[:, :rank].T
    param_dim = rank
    
    # CMA-ES initial parameters
    init_params = np.zeros(param_dim)  # start from zero matrix
    cma_options = {
        "seed": seed,
        "popsize": population_size,
        "maxiter": num_iterations,
        "verb_disp": 1,  # Print updates
        "bounds": None,  # We won't clamp parameters
    }
    
    # Create CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(init_params, init_sigma, cma_options)
    
    best_params = None
    best_score = -np.inf

    iteration = 0

    log_file = open(f"logs/cma_es_halfcheetah_rndm_prj_log_rank_{rank}_take_{exp_id}.txt", "w")
    # CMA-ES main loop
    while not es.stop():
        iteration += 1
        
        # Ask CMA-ES for candidate solutions
        solutions = es.ask()
        
        # Evaluate each candidate solution
        rewards = []
        for sol in solutions:
            # Evaluate on environment
            # sol_low = sol
            # sol_low_to_high = sol_low @ low_to_high_projection_matrix
            # sol_low_to_high_to_low = sol_low_to_high @ high_to_low_projection_matrix
            # print(sol_low, sol_low_to_high_to_low)
            # exit()
            sol = sol @ low_to_high_projection_matrix
            reward = evaluate_policy(env, sol, 
                                     num_episodes=rollout_episodes,
                                     max_steps=max_steps)
            rewards.append(-reward)  
            # CMA-ES is a minimizer by default, 
            # so we use the negative of the reward as the "cost"

            # Track best solution
            if reward > best_score:
                best_score = reward
                best_params = sol.copy()

        # Tell CMA-ES about our solutions' negative rewards
        es.tell(solutions, rewards)
        
        # Log current status
        print(f"Iteration {iteration}/{num_iterations} "
              f"| Best Reward So Far: {best_score:.2f} "
              f"| Mean Reward This Iter: {(-np.mean(rewards)):.2f}")
        log_file.write(f"Mean Reward: {(-np.mean(rewards)):.2f}\n")
        log_file.flush()
        
        if iteration >= num_iterations:
            log_file.close()
            break

    env.close()
    
    # Final best solution
    return best_params, best_score


def train_hopper_cma_es(exp_id, rank):
    # Hyperparameters (tune as needed)
    ENV_NAME = "HalfCheetah-v5"
    SEED = np.random.randint(0, 10000)
    NUM_ITERATIONS = 400  # More iterations can improve performance
    POP_SIZE = 16        # Default "popsize" for CMA-ES is ~4+3*log(dim), 
                         # you can increase or decrease
    INIT_SIGMA = 0.5
    ROLLOUT_EPISODES = 1
    MAX_STEPS = 1000

    best_params, best_return = train_linear_policy_cma_es(
        env_name=ENV_NAME,
        seed=SEED,
        num_iterations=NUM_ITERATIONS,
        population_size=POP_SIZE,
        init_sigma=INIT_SIGMA,
        rollout_episodes=ROLLOUT_EPISODES,
        max_steps=MAX_STEPS,
        exp_id=exp_id,
        rank=rank,
    )

    print("\n======== CMA-ES Training Finished ========")
    print("Best flattened parameters:\n", best_params)
    print("Best Weight Matrix (W) of shape (8,2):\n", best_params.reshape(OBS_DIM, ACT_DIM))
    print(f"Best observed return: {best_return:.2f}")





import gymnasium as gym
import numpy as np
import cma
import sys


def evaluate_linear_policy_with_bias(env, params, num_episodes=1, max_steps=1000):
    """
    Evaluate a linear policy (action = W @ obs + b) on Hopper-v5.
    
    params: 1D np.array of length 36 (since 3*11 + 3 = 36)
            - The first 33 elements are W (3x11)
            - The last 3 elements are b (3,)
    Returns the average total reward over num_episodes.
    """
    # Extract W and b from flattened params
    # W shape: (3, 11), b shape: (3,)
    W_flat = params[:OBS_DIM * ACT_DIM]
    b_flat = params[OBS_DIM * ACT_DIM:]
    
    W = W_flat.reshape(ACT_DIM, OBS_DIM)
    b = b_flat.reshape(ACT_DIM,)  # or just b_flat

    total_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            # Compute action = W @ obs + b
            action = W.dot(obs) + b
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def train_linear_policy_cma_es_with_bias(env_name="Hopper-v5",
                                         seed=123,
                                         max_generations=10,
                                         population_size=16,
                                         init_sigma=0.5,
                                         rollout_episodes=1,
                                         max_steps=1000,
                                         exp_id=0,
                                         rank=5):
    """
    Train a linear-with-bias policy using CMA-ES on Hopper-v5.
    
    Policy parameters: length 36 = (3*11 + 3)
      - first 3*11=33 for W (3x11)
      - last 3 for b
    """
    np.random.seed(seed)

    # Create Hopper-v5 environment
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]  # 11 for Hopper-v5
    act_dim = env.action_space.shape[0]       # 3 for Hopper-v5
    global OBS_DIM, ACT_DIM
    OBS_DIM = obs_dim
    ACT_DIM = act_dim

    param_dim = act_dim * obs_dim + act_dim  # 3*11 + 3 = 36
    G = np.random.randn(param_dim, param_dim)
    Q, R = np.linalg.qr(G)
    high_to_low_projection_matrix = Q[:, :rank]
    low_to_high_projection_matrix = Q[:, :rank].T
    param_dim = rank

    # CMA-ES options
    init_params = np.zeros(param_dim)   # Start from all zeros
    cma_options = {
        "seed": seed,
        "popsize": population_size,  # number of solutions each generation
        "maxiter": max_generations,  # maximum number of generations
        "verb_disp": 1,             # print CMA-ES progress
        # You could specify "bounds" if you'd like to clamp parameters
        # e.g. "bounds": [ -5, 5 ],
    }

    # Create the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(init_params, init_sigma, cma_options)

    best_params = None
    best_score = -np.inf
    generation = 0

    log_file = open(f"logs/cma_es_bias_halfcheetah_rndm_prj_log_rank_{rank}_take_{exp_id}.txt", "w")
    while not es.stop():
        generation += 1
        # Ask CMA-ES for candidate solutions
        solutions = es.ask()

        # Evaluate each solution
        fitness_list = []
        for sol in solutions:
            # CMA-ES is a minimizer, so we'll feed negative reward as the "cost"
            sol = sol @ low_to_high_projection_matrix
            reward = evaluate_linear_policy_with_bias(
                env, sol, num_episodes=rollout_episodes, max_steps=max_steps
            )
            cost = -reward
            fitness_list.append(cost)

            # Track the best
            if reward > best_score:
                best_score = reward
                best_params = sol.copy()

        # Update CMA-ES with the fitness for each solution
        es.tell(solutions, fitness_list)

        print(f"Generation {generation}/{max_generations} "
              f"| Best Reward So Far: {best_score:.2f} "
              f"| Mean Reward This Gen: {-np.mean(fitness_list):.2f}")

        log_file.write(f"Mean Reward: {(-np.mean(fitness_list)):.2f}\n")
        log_file.flush()

        # If we've reached max_generations, we can stop
        if generation >= max_generations:
            log_file.close()
            break

    env.close()
    return best_params, best_score


def train_hopper_cma_es_with_bias(exp_id, rank):
    # Hyperparameters
    ENV_NAME = "HalfCheetah-v5"
    SEED = np.random.randint(0, 10000)
    MAX_GENERATIONS = 400
    POP_SIZE = 64
    INIT_SIGMA = 0.5
    ROLLOUT_EPISODES = 1
    MAX_STEPS = 1000

    # Train
    best_params, best_return = train_linear_policy_cma_es_with_bias(
        env_name=ENV_NAME,
        seed=SEED,
        max_generations=MAX_GENERATIONS,
        population_size=POP_SIZE,
        init_sigma=INIT_SIGMA,
        rollout_episodes=ROLLOUT_EPISODES,
        max_steps=MAX_STEPS,
        exp_id=exp_id,
        rank=rank,
    )

    # Print final results
    print("\n======== CMA-ES (Linear + Bias) Training Finished ========")
    print(f"Best observed return: {best_return:.2f}")
    # print("Best flattened parameters shape:", best_params.shape)  
    # # print("W shape = (3,11), b shape = (3,)")
    # W_flat = best_params[:OBS_DIM * ACT_DIM]
    # b_flat = best_params[OBS_DIM * ACT_DIM:]
    # print("W:\n", W_flat.reshape(OBS_DIM, ACT_DIM))
    # print("b:\n", b_flat)






if __name__ == "__main__":
    # train_hopper_cem()
    # train_hopper_cem_bias()

    # add a argument to the function, which is an integer, from the command line
    if len(sys.argv) != 3:
        print("Usage: python train_hopper_random_projection.py exp_id rank")
        sys.exit(1)

    exp_id = int(sys.argv[1])
    rank = int(sys.argv[2])

    # train_hopper_cma_es(exp_id, rank)
    train_hopper_cma_es_with_bias(exp_id, rank)