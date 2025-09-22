import gymnasium as gym
import numpy as np
import os

def make_world(name, video_prefix):
    env = gym.make(name, render_mode="rgb_array")

    #env = gym.wrappers.RecordVideo(
    #    env = env,
    #    video_folder = "logs/video",
    #    name_prefix = video_prefix,
    #    episode_trigger = lambda e_idx: True
    #)

    return env


def execute(env, action):
    state, reward, done, _, _ = env.step(action)

    return state, reward, done


def linear_policy(weights):
    def get_action(state):
        return np.matmul(state, weights[:-1]) + weights[-1]

    return get_action


# def run(iteration, name, weights):
#     print(f"Starting episode number {iteration + 1}")
#     env = make_world(name, iteration)
#     state, _ = env.reset()
#     policy = linear_policy(weights)
#
#     #env.start_video_recorder()
#     done = False
#     truncated = False
#     reward = 0
#     step = 1
#     current_dir = f"logs/video/cartpole_3_{iteration + 1}"
#     os.makedirs(current_dir, exist_ok = True)
#     while not (done or truncated):
#         action = policy(state).argmax()
#         state, _reward, done, truncated, _ = env.step(action)
#         img = env.render()
#         cv2.imshow(name, img)
#         cv2.waitKey(1)
#         cv2.imwrite(f'{current_dir}/{step}.png', img)
#
#         reward += _reward
#         step += 1
#
#     #env.close_video_recorder()
#     env.close()
#     print("Final reward =", reward)

def run_and_record(env_name, weights, save_dir="vid"):
    # Create environment with video recording
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=save_dir,
        episode_trigger=lambda episode_id: True  # Record every episode (we run one)
    )

    state, _ = env.reset()
    policy = linear_policy(weights)
    done = False
    truncated = False
    total_reward = 0

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    while not (done or truncated):
        action_scores = policy(state)
        if discrete:
            action = int(np.argmax(action_scores))
        else:
            action = np.array(action_scores, dtype=np.float32).reshape(env.action_space.shape)

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    print("Final reward:", total_reward)

def experiments(number, name, weights):
    print(f"Starting {number} experiments")
    for i in range(number):
        run_and_record(name, weights, save_dir=f"logs/{name}_video_best_policy")


if __name__ == "__main__":
    name = "CartPole-v1"
    # this has 500 value
    cartpole_weights = [[-0.05, 0.05], [-0.25, 0.25], [-2.0, 2.0], [-2.5, 2.5], [0.05, 0.06]]
    # this has 1000 value
    ip_weights = [[0.4], [6.0], [0.6], [2.5], [0.0]]
    mc_weights = [[0.0],	[5.5],	[0.0]]
    pong_weights = [[3.5, -2.5, 1.0], [-1.0, 0.5, 2.0], [-1.5, 4.0, -3.0], [4.0, -4.0, 5.0], [-5.0, 6.0, -6.0],
                    [1.5, -1.5, 3.0]]
    swimmer_weights = [[4.5, -1.5], [4.5, 5.0], [-2.0, 4.0], [1.0, -2.5], [2.0, 5.0], [-3.0, 3.5], [-1.0, 4.5],
                       [-4.0, 2.5], [3.0, -2.5]]
    # weights = [[0.1, 0.2], [0.2, 0.3], [0.6, 1.5], [0.4, 0.9], [0.05, 0.1]]
    # weights = np.array([0.1, 0.2, 0.6, 0.4, 0.05])
    _num = 1
    experiments(_num, name, cartpole_weights)
    experiments(_num, "Swimmer-v5", swimmer_weights)
    experiments(_num, "InvertedPendulum-v5", ip_weights)
    experiments(_num, "MountainCarContinuous-v0", mc_weights)
    # experiments(_num, "PongNoFrameskip-v5", pong_weights)
