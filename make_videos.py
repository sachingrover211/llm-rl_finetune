import gymnasium as gym
import numpy as np
import cv2, os

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



def run(iteration, name, weights):
    print(f"Starting episode number {iteration + 1}")
    env = make_world(name, iteration)
    state, _ = env.reset()
    policy = linear_policy(weights)

    #env.start_video_recorder()
    done = False
    truncated = False
    reward = 0
    step = 1
    current_dir = f"logs/video/cartpole_3_{iteration + 1}"
    os.makedirs(current_dir, exist_ok = True)
    while not (done or truncated):
        action = policy(state).argmax()
        state, _reward, done, truncated, _ = env.step(action)
        img = env.render()
        cv2.imshow(name, img)
        cv2.waitKey(1)
        cv2.imwrite(f'{current_dir}/{step}.png', img)

        reward += _reward
        step += 1

    #env.close_video_recorder()
    env.close()
    print("Final reward =", reward)


def experiments(number, name, weights):
    print(f"Starting {number} experiments")
    for i in range(number):
        run(i, name, weights)


if __name__ == "__main__":
    name = "CartPole-v1"
    # this has 500 value
    #weights = [[-0.05, 0.05], [-0.25, 0.25], [-2.0, 2.0], [-2.5, 2.5], [0.05, 0.06]]
    weights = [[0.1, 0.2], [0.2, 0.3], [0.6, 1.5], [0.4, 0.9], [0.05, 0.1]]
    _num = 20

    experiments(_num, name, weights)
