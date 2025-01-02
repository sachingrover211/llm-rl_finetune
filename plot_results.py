import numpy as np
import os

all_succ = []
# root_folder = 'logs/swimmer_continuous_sas_v1_100'
# root_folder = 'logs/mountaincar_continuous_sas_v1_100'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v1_100'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v2_30'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v3_30'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v4_30'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v4_2_30'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v3_2_400'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v4_3_400'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v5_400'
# root_folder = 'logs/swimmer_continuous_random_search_0.1'
# root_folder = 'logs/swimmer_continuous_random_search_0.05'
# root_folder = 'logs/swimmer_continuous_random_search_0.01'
# root_folder = 'logs/swimmer_continuous_random_search_0.005'
# root_folder = 'logs/swimmer_continuous_random_search_0.1_take2'
# root_folder = 'logs/swimmer_continuous_random_search_0.05_take2'
# root_folder = 'logs/swimmer_continuous_random_search_0.01_take2'
# root_folder = 'logs/swimmer_continuous_random_search_0.005_take2'
# root_folder = 'logs/swimmer_continuous_mean_std_sas_v5_400_std_0.01'
# root_folder = 'logs/swimmer_continuous_llm_num_optim_400'
root_folder = 'logs/swimmer_continuous_llm_num_optim_400_std_0.01'
all_folders = [os.path.join(root_folder, x) for x in os.listdir(root_folder) if 'episode' in x]
all_folders.sort(key=lambda x: int(x.split('_')[-1]))
for folder in all_folders:
    print(folder)
    # folder = f'mountain_car_logs/non_explainable/{training_episode + 1}'
    # folder = f'logs/swimmer_continuous_sas_v1_100/episode_{training_episode}'
    # read all text files in the folder. Read the last line of each file and extract the total reward. The last line looks like this: "Total reward: -157.0"
    rewards_succ = []
    rewards_fail = []
    for filename in os.listdir(folder):
        if 'evaluation' in filename:
        # if True:
            with open(os.path.join(folder, filename), 'r') as f:
                lines = f.readlines()
                try:
                    last_line = lines[-1]
                    total_reward = float(last_line.split()[-1])
                except:
                    last_line = lines[-2]
                    total_reward = float(last_line.split()[-1])
                if total_reward > -300:
                    rewards_succ.append(total_reward)
                else:
                    rewards_fail.append(total_reward)
    print(rewards_succ)
    print(rewards_fail)
    
    all_rewards = rewards_succ + rewards_fail

    # print("Average reward for successful episodes:", np.mean(rewards_succ))
    # print("Standard deviation of reward for successful episodes:", np.std(rewards_succ))
    # print("Average reward for failed episodes:", np.mean(rewards_fail))
    # print("Standard deviation of reward for failed episodes:", np.std(rewards_fail))
    print("Average reward for all episodes:", np.mean(all_rewards))
    print("Standard deviation of reward for all episodes:", np.std(all_rewards))
    all_succ.append(np.mean(all_rewards))
print(all_succ)


all_succ = [x if x > -100 else None for x in all_succ]

import matplotlib.pyplot as plt

# Data for the plot

episodes = list(range(1, len(all_succ) + 1))

# Creating the plot
plt.figure(figsize=(8, 6))
# plt.plot(episodes, rewards_1, label="Q-Learning with LLM", marker='o')
# plt.plot(episodes[:5], rewards_2, label="Q-Learning with LLM and human advice", marker='s')
# plt.plot(episodes, all_succ, label="Linear Policy Update with LLM", marker='s')
plt.plot(episodes, all_succ, label="Linear RL with LLM", marker='s')

# Adding labels, legend, and title
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title(root_folder)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
