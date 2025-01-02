import os
import numpy as np


def load_parameters_from_episode(episode_folder):
    # Load parameters from episode folder
    params_file = os.path.join(episode_folder, 'parameters.txt')
    params = []
    with open(params_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                params.append([float(x) for x in line.strip().split(',')])
            except:
                pass
        f.close()
    return np.array(params)


def l2_distance(params1, params2):
    # Compute L2 distance between two sets of parameters
    return np.linalg.norm(params1 - params2)


def compare_parameters(all_params):
    # Compare parameters from all episodes
    print(len(all_params))
    window = all_params[:20]
    distances = []
    for i in range(20, len(all_params)):
        distances.append(l2_distance(all_params[i], np.mean(window, axis=0)))
        window.pop(0)
        window.append(all_params[i])
    return distances

def compare_parameters_min(all_params):
    # Compare parameters from all episodes
    print(len(all_params))
    window = all_params[:20]
    distances = []
    for i in range(20, len(all_params)):
        min_distance = np.inf
        for j in range(20):
            if l2_distance(all_params[i], window[j]) < min_distance:
                min_distance = l2_distance(all_params[i], window[j])
        distances.append(min_distance)
        window.pop(0)
        window.append(all_params[i])
    return distances

def compare_parameters_min_all_history(all_params):
    # Compare parameters from all episodes
    print(len(all_params))
    window = all_params[:20]
    distances = []
    for i in range(20, len(all_params)):
        min_distance = np.inf
        for j in range(len(window)):
            if l2_distance(all_params[i], window[j]) < min_distance:
                min_distance = l2_distance(all_params[i], window[j])
        distances.append(min_distance)
        window.append(all_params[i])
    return distances


def compare_parameters_min_first_20(all_params):
    # Compare parameters from all episodes
    print(len(all_params))
    window = all_params[:20]
    distances = []
    for i in range(20, len(all_params)):
        min_distance = np.inf
        for j in range(len(window)):
            if l2_distance(all_params[i], window[j]) < min_distance:
                min_distance = l2_distance(all_params[i], window[j])
        distances.append(min_distance)
    return distances

if __name__ == '__main__':
    # Define root log folder
    # root_folder = 'logs/swimmer_continuous_mean_std_sas_v5_400'
    # root_folder = 'logs/swimmer_continuous_mean_std_sas_v4_3_400'
    root_folder = 'logs/swimmer_continuous_mean_std_sas_v3_2_400'

    # Find all episode folders
    all_folders = [os.path.join(root_folder, x) for x in os.listdir(root_folder) if x.startswith('episode')]
    all_folders.sort(key=lambda x: int(x.split('_')[-1]))
    all_folders = all_folders[:-1]

    # Load parameters from all episodes
    all_params = []
    for folder in all_folders:
        all_params.append(load_parameters_from_episode(folder))

    # Compare parameters from all episodes
    # results = compare_parameters(all_params)
    # results = compare_parameters_min(all_params)
    # results = compare_parameters_min_all_history(all_params)
    results = compare_parameters_min_first_20(all_params)

    # Plot results
    import matplotlib.pyplot as plt

    plt.plot(range(20, len(results) + 20), results)
    plt.xlabel('Episode')
    plt.ylabel('L2 distance')
    plt.title('L2 distance between consecutive episodes')
    plt.show()

