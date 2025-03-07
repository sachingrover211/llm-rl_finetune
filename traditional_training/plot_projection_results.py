import numpy as np
from scipy.stats import ttest_rel, wilcoxon

root = 'logs'

def read_experiment_results(name):
    scores = []
    for i in range(10):
        curr_scores = []
        with open(f'{root}/{name}_take_{i+1}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                score = float(line.split()[-1])
                curr_scores.append(score)
        scores.append(curr_scores)
    return np.array(scores)

def find_best_exp_results(scores):
    best_scores = []
    for i, curr_scores in enumerate(scores):
        best_scores.append(max(curr_scores))
    return best_scores


def find_lower_upper_bound(name):
    if 'swimmer' in name:
        return -100, 1000
    elif 'hopper' in name:
        return -100, 3000
    elif 'halfcheetah' in name:
        return -3000, 3000
    else:
        return -100, 1000


# Suppose these contain performance metrics across repeated runs:

B_name = 'cma_es_swimmer_rndm_prj_log_rank_5'
B_name = 'cma_es_halfcheetah_rndm_prj_log_rank_5'
B_name = 'cma_es_bias_halfcheetah_rndm_prj_log_rank_5'
lower_lim, upper_lim = find_lower_upper_bound(B_name)



raw_scores_B = read_experiment_results(B_name)
scores_B = find_best_exp_results(raw_scores_B)

# Use matplotlib to plot the raw scores. Find the std deviation of the scores. Use 2 colors to plot the 2 sets of scores.
import matplotlib.pyplot as plt

plt.figure()
raw_scores_B = np.array(raw_scores_B)
std_B = np.std(raw_scores_B, axis=0)
mean_B = np.mean(raw_scores_B, axis=0)

# use fill_between to show the standard deviation
# plt.plot(np.arange(len(std_A)), mean_A, label='PCA Projection', color='blue')
# plt.fill_between(np.arange(len(std_A)), mean_A - std_A, mean_A + std_A, color='blue', alpha=0.2)
plt.plot(np.arange(len(std_B)), mean_B, label='Random Projection', color='red')
plt.fill_between(np.arange(len(std_B)), mean_B - std_B, mean_B + std_B, color='red', alpha=0.2)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.ylim(lower_lim, upper_lim)
plt.legend()
# plt.legend(loc='lower right')
plt.title(B_name)
plt.show()


fig, axs = plt.subplots(5, 2, figsize=(15, 20))
axs = axs.flatten()

for i in range(10):
    axs[i].plot(np.arange(len(std_B)), raw_scores_B[i], label=f"Run {i+1}", c='red')
    axs[i].set_xlabel("Episodes")
    axs[i].set_ylabel("Reward")
    axs[i].set_title(f"Run {i+1}")
    axs[i].set_ylim(lower_lim, upper_lim)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
# plt.savefig(f'results_curves/random_beam_search_subplots.png')
plt.show()