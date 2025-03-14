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
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_random'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_2' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_3' # v2
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_4' # v2
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_5' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_take_6' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gemini_1.5_flash_2_take_1' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gpt-4o-mini_2_take_2' # v1
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gpt-4o-mini_more_hist_2_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gpt-4o-mini_3_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gpt-4o-mini_3_forward_reward_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_temp_1.2_gpt-4o-mini_3_forward_reward_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_forward_reward_only_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_forward_reward_only_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_forward_reward_only_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_take_4'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_take_5'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_take_1' # original reward with llm to supply the randomness in beam search
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_take_3'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_take_4'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_take_5'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_biased_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_biased_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_biased_take_3'

# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_take_2'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_reward_take_1'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_reward_take_2_pretrained'
# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_llm_reward_take_3_pretrained_denser_traj'

# root_folder = 'logs/mujoco_hopper_llm_num_optim_400_no_bias_range_std_2_expected_r_beam_3_random_forward_reward_only_take_3'

# root_folder = 'logs/panda_reach_llm_num_optim_300_no_bias_range'
# root_folder = 'logs/panda_reach_llm_num_optim_300_no_bias_std_1_range'

# root_folder = 'logs/cliff_walking_v2_o3_mini_take_1'
# root_folder = 'logs/cliff_walking_v2_4o_take_1'
# root_folder = 'logs/cliff_walking_v2_4o_take_2_new_template'
# root_folder = 'logs/cliff_walking_v2_o3_mini_new_template_3' # good
# root_folder = 'logs/cliff_walking_v2_4o_new_template_4'


# root_folder = 'logs/maze3x3_o3_mini_new_template_5'
# root_folder = 'logs/maze3x3_o3_mini_new_template_7' # good
# root_folder = 'logs/maze5x5_o3_mini_new_template_7'
# root_folder = 'logs/maze5x5_o3_mini_new_template_7_take_2'
# root_folder = 'logs/maze5x5_o3_mini_new_template_7_take_3'

# root_folder = 'logs/pong_llm_num_optim_300_no_bias_std_1'
# root_folder = 'logs/panda_reach_llm_num_optim_300_no_bias_std_1_range_6d'
# root_folder = 'logs/panda_reach_llm_num_optim_300_no_bias_std_1_range_6d_sparse'

# root_folder = 'logs/nav_track_llm_num_optim_300_no_bias_std_1'
# root_folder = 'logs/nav_track_llm_num_optim_300_no_bias_std_1_exp_o3_mini'
# root_folder = 'logs/nav_track_llm_num_optim_300_no_bias_std_1_exp'

# root_folder = 'logs/frozen_lake_o3_mini_8'
# root_folder = 'logs/frozen_lake_4o_reflex' # good
# root_folder = 'logs/frozen_lake_4o_reflex_10_training_rollouts'

# # Nim
# root_folder = 'logs/nim_4o_reflex_10_training_rollouts' # good

# # Mountain Car Continuous with Imitation Learning
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_no_bias_range_imitation'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_no_bias_range_imitation_take_2'
# root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_no_bias_range_imitation_take_3'

# # Blackjack
# root_folder = 'logs/blackjack_v1_take_2_500_warmup'
# root_folder = 'logs/blackjack_v1_take_2_500_warmup_best_1_q_tables'

# root_folder = 'logs/mountaincar_llm_num_optim_q_table_400_rank_5_take_1'
# root_folder = 'logs/mountaincar_llm_num_optim_q_table_400_rank_10_take_1'
# root_folder = 'logs/mountaincar_llm_num_optim_q_table_400_rank_15_take_1'
# root_folder = 'logs/mountaincar_llm_num_optim_q_table_400_rank_20_take_1'
# root_folder = 'logs/cliffwalking_llm_num_optim_q_table_400_rank_5_take_1'
# root_folder = 'logs/nim_llm_num_optim_q_table_400_rank_5_take_1'
root_folder = 'logs/mountaincar_continuous_action_llm_num_optim_400_std_1_expected_r_no_bias_range_semantics'

all_folders = [os.path.join(root_folder, x) for x in os.listdir(root_folder) if 'episode' in x]
all_folders.sort(key=lambda x: int(x.split('_')[-1]))
for folder in all_folders:
    print(folder)
    # folder = f'mountain_car_logs/non_explainable/{training_episode + 1}'
    # folder = f'logs/swimmer_continuous_sas_v1_100/episode_{training_episode}'
    # read all text files in the folder. Read the last line of each file and extract the total reward. The last line looks like this: "Total reward: -157.0"
    rewards_succ = []
    rewards_fail = []
    if 'blackjack' not in root_folder:
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
    else:
        curr_episode_rewards = []
        for filename in os.listdir(folder):
            if 'evaluation' in filename:
                with open(os.path.join(folder, filename), 'r') as f:
                    lines = f.readlines()
                    curr_rewards = []
                    for line in lines[1:]:
                        curr_rewards.append(float(line.split('|')[-1]))
                    curr_rewards = np.sum(curr_rewards)
                    curr_episode_rewards.append(curr_rewards)
        rewards_succ.append(np.mean(curr_episode_rewards))
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


global_optimum = None
if 'hopper' in root_folder.lower():
    global_optimum = 2300
elif 'mountain' in root_folder.lower():
    global_optimum = 100
elif 'cliff' in root_folder.lower():
    global_optimum = -13


# all_succ = [x if x > -500 and x < 200 else None for x in all_succ]
# all_succ = [x if x < 200 else None for x in all_succ]

import matplotlib.pyplot as plt

# Data for the plot

episodes = list(range(1, len(all_succ) + 1))
if root_folder == 'logs/frozen_lake_4o_reflex':
    episodes = [20 * x for x in episodes]

    episodes = episodes[:150]
    all_succ = all_succ[:150]
elif root_folder == 'logs/frozen_lake_4o_reflex_10_training_rollouts':
    episodes = [10 * x for x in episodes]

    episodes = episodes[:300]
    all_succ = all_succ[:300]

# Creating the plot
plt.figure(figsize=(8, 6))
# plt.plot(episodes, rewards_1, label="Q-Learning with LLM", marker='o')
# plt.plot(episodes[:5], rewards_2, label="Q-Learning with LLM and human advice", marker='s')
# plt.plot(episodes, all_succ, label="Linear Policy Update with LLM", marker='s')
plt.plot(episodes, all_succ, label="Linear RL with LLM", marker='s')

if global_optimum is not None:
    plt.axhline(y=global_optimum, color='r', linestyle='--', alpha=0.5, label='Global Optimum')

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
