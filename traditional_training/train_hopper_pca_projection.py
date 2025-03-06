import gymnasium as gym
import numpy as np
from sklearn.decomposition import PCA
import sys


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
            W = W.reshape(11, 3)
            action = np.dot(obs, W)  # shape (3,)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)



import cma  # pip install cma



def train_linear_policy_cma_es(env_name="Hopper-v5",
                               seed=123,
                               num_iterations=10,
                               population_size=16,
                               init_sigma=0.5,
                               rollout_episodes=1,
                               max_steps=1000,
                               exp_id=0):
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
    assert obs_dim == 11, f"Expected obs_dim=11, got {obs_dim}"
    assert act_dim == 3,  f"Expected act_dim=3, got {act_dim}"
    
    # Flatten dimension for linear W
    rank = 10
    param_dim = obs_dim * act_dim  # 11*3=33


    # Create random projection matrix
    random_policies = np.random.randn(100, param_dim)
    pca = PCA(n_components=rank)
    pca.fit(random_policies)

    
    # CMA-ES initial parameters
    init_params = np.zeros(rank)  # start from zero matrix
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

    log_file = open(f"logs/cma_es_hopper_pca_prj_rank_10_log_take_{exp_id}.txt", "w")
    # CMA-ES main loop
    while not es.stop():
        iteration += 1
        
        # Ask CMA-ES for candidate solutions
        solutions = es.ask()
        
        # Evaluate each candidate solution
        rewards = []
        for sol in solutions:
            # Evaluate on environment
            sol = pca.inverse_transform(sol)#.reshape(obs_dim, act_dim)
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
        
        if iteration >= num_iterations:
            log_file.close()
            break

    env.close()
    
    # Final best solution
    return best_params, best_score


def train_hopper_cma_es(arg):
    # Hyperparameters (tune as needed)
    ENV_NAME = "Hopper-v5"
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
        exp_id=arg,
    )

    print("\n======== CMA-ES Training Finished ========")
    print("Best flattened parameters:\n", best_params)
    print("Best Weight Matrix (W) of shape (11,3):\n", best_params.reshape(11, 3))
    print(f"Best observed return: {best_return:.2f}")






if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_hopper_random_projection.py <integer>")
        sys.exit(1)

    arg = int(sys.argv[1])
    train_hopper_cma_es(arg)