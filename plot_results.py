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
# root_folder = 'logs/swimmer_continuous_llm_num_optim_400_std_0.01'
# root_folder = 'logs/swimmer_continuous_llm_num_optim_400_std_0.01_removed_range'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_0.1'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_2'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_3'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_4'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_5'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_6'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_take_7'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_shifted_parameters'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_range_shifted_parameters_explanation'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_2'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_3'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_4'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_5'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_6'
# root_folder = 'logs/mujoco_invertedpendulum_llm_num_optim_300_no_bias_range_std_2_expected_r_take_7'
# root_folder = 'logs/pendulum_llm_num_optim_300_no_bias_range_std_1_expected_r_iter'
# root_folder = 'logs/pendulum_llm_num_optim_300_no_bias_range_std_1_expected_r_iter_reduced_process'
# root_folder = 'logs/pendulum_llm_num_optim_300_no_bias_range_std_1_expected_r_iter_low_gravity_6_reduced_process'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_no_bias_range_std_1_expected_r_iter_reduced_process'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_1000_no_bias_range_std_1_expected_r_iter_reduced_process_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_1000_no_bias_range_std_1_expected_r_iter_reduced_process_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math_llm'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math_llm_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math_llm_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_300_improved_std_range_no_bias_iterative_params_math_llm_4_o1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_change_expected_r_take_2'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_no_bias_norm'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_norm'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_norm_take_2'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_norm_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_std_change_expected_r_norm_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_std_change_expected_r_norm_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_small_expected_r_elite_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_change_expected_r_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_expected_r_delta_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_delta_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_expected_r_4o_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_expected_r_gemini_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_expected_r_delta_4o_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_4o_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_expected_r_delta_gemini_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_gemini_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_gemini_descent_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_gemini_descent_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_gemini_descent_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_adaptive_true_expected_r_delta_gemini_descent_reverse_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_change_expected_r_take_4'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_3'
root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_4'
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
        # if 'evaluation' in filename:
        if 'training' in filename:
        # if True:
            with open(os.path.join(folder, filename), 'r') as f:
                lines = f.readlines()
                rewards = []
                for line in lines:
                    if 'Total reward' in line:
                        total_reward = float(line.split()[-1])
                        rewards.append(total_reward)
                rewards_succ.append(np.mean(rewards))
    print(rewards_succ)
    print(rewards_fail)
    
    all_rewards = rewards_succ + rewards_fail

    # print("Average reward for successful episodes:", np.mean(rewards_succ))
    # print("Standard deviation of reward for successful episodes:", np.std(rewards_succ))
    # print("Average reward for failed episodes:", np.mean(rewards_fail))
    # print("Standard deviation of reward for failed episodes:", np.std(rewards_fail))
    print("Average reward for all episodes:", np.mean(all_rewards))
    print("Standard deviation of reward for all episodes:", np.std(all_rewards))

    if 'descent' in root_folder:
        all_succ.append(1500 - np.mean(all_rewards))
    else:
        all_succ.append(np.mean(all_rewards))
print(all_succ)
print(max(all_succ))
for i in range(len(all_succ)):
    if all_succ[i] >= max(all_succ) * 0.95:
        print(i + 1)
        break


# all_succ = [x if x > -500 and x < 200 else None for x in all_succ]
# all_succ = [x if x < 200 else None for x in all_succ]

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
plt.savefig(f'results_curves/{root_folder}.png')
plt.show()
