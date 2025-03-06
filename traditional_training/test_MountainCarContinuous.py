import gymnasium as gym
import numpy as np
import os
import json

# Define the linear policy matrix (11 x 3)
W = np.array([-0.35, 7.0]).reshape(2, 1)

print("W shape:", W.shape)
print("W:\n", W)


def evaluate_linear_policy(env, W, log_dir, num_episodes=3, max_steps=1000):
    """
    Evaluate a linear policy (action = obs @ W) on Hopper-v5.
    Renders each step so a human can watch.
    """
    all_total_rewards = []
    os.makedirs(log_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        state_action_log = []
        for t in range(max_steps):
            # Compute action from current observation
            action = obs @ W  # shape: (3, )
            # action = obs @ W + b  # shape: (3, )

            state_action_log.append({"state": [float(x) for x in list(obs)], "action": [float(x) for x in list(action)]})

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Render for a human to see
            # env.render()

            total_reward += reward
            if terminated or truncated:
                print(
                    f"Episode {ep+1} ended at step {t+1} with total reward: {total_reward:.2f}"
                )
                # with open(os.path.join(log_dir, f"episode_{ep+1}.json"), "w") as f:
                #     json.dump(state_action_log, f, indent=4)
                break

        else:
            # If we exited by max_steps
            print(
                f"Episode {ep+1} reached max steps ({max_steps}) with total reward: {total_reward:.2f}"
            )

        all_total_rewards.append(total_reward)
    return all_total_rewards


if __name__ == "__main__":
    # Create Hopper-v5 environment with human-render mode
    env = gym.make("MountainCarContinuous-v0", render_mode='human')
    # env = gym.make("Hopper-v5", render_mode=None)

    # Evaluate the matrix W for multiple episodes
    all_total_rewards = evaluate_linear_policy(
        env,
        W,
        log_dir="results_curves/MountainCarContinuous",
        num_episodes=10,
        max_steps=1000,
    )
    print("Mean total reward:", np.mean(all_total_rewards))

    # Close the environment once finished
    env.close()
