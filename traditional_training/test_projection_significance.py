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

# Suppose these contain performance metrics across repeated runs:

# A_name = 'cma_es_hopper_pca_prj_log'
# B_name = 'cma_es_hopper_rndm_prj_log'
# A_name = 'cma_es_hopper_pca_prj_rank_10_log'
# B_name = 'cma_es_hopper_rndm_prj_log_rank_10'
A_name = 'cma_es_swimmer_rndm_prj_log_rank_5'
B_name = 'cma_es_swimmer_rndm_prj_log_rank_5'

# B_name = 'cma_es_hopper_rndm_prj_log_rank_5'
B_name = 'cma_es_swimmer_rndm_prj_log_rank_5'
raw_scores_A = read_experiment_results(A_name)
raw_scores_B = read_experiment_results(B_name)
scores_A = find_best_exp_results(raw_scores_A)
scores_B = find_best_exp_results(raw_scores_B)

# Paired t-test
t_stat, p_value = ttest_rel(scores_A, scores_B)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}")

# Wilcoxon signed-rank test
w_stat, p_value_wil = wilcoxon(scores_A, scores_B)
print(f"Wilcoxon signed-rank test: statistic = {w_stat}, p-value = {p_value_wil}")

# Use matplotlib to plot the raw scores. Find the std deviation of the scores. Use 2 colors to plot the 2 sets of scores.
import matplotlib.pyplot as plt

plt.figure()
raw_scores_A = np.array(raw_scores_A)
raw_scores_B = np.array(raw_scores_B)
print(raw_scores_A.shape)
std_A = np.std(raw_scores_A, axis=0)
std_B = np.std(raw_scores_B, axis=0)
mean_A = np.mean(raw_scores_A, axis=0)
mean_B = np.mean(raw_scores_B, axis=0)

print(std_A.shape)

# use fill_between to show the standard deviation
# plt.plot(np.arange(len(std_A)), mean_A, label='PCA Projection', color='blue')
# plt.fill_between(np.arange(len(std_A)), mean_A - std_A, mean_A + std_A, color='blue', alpha=0.2)
plt.plot(np.arange(len(std_A)), mean_B, label='Random Projection', color='red')
plt.fill_between(np.arange(len(std_A)), mean_B - std_B, mean_B + std_B, color='red', alpha=0.2)
plt.xlabel('Episodes')
plt.ylabel('Reward for hopper')
plt.legend()
# plt.legend(loc='lower right')
plt.title('Swimmer; Rank=5')
plt.show()


fig, axs = plt.subplots(5, 2, figsize=(15, 20))
axs = axs.flatten()

for i in range(10):
    axs[i].plot(np.arange(len(std_A)), raw_scores_B[i], label=f"Run {i+1}", c='red')
    axs[i].set_xlabel("Episodes")
    axs[i].set_ylabel("Reward")
    axs[i].set_title(f"Run {i+1}")
    axs[i].set_ylim(-120, 1200)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
# plt.savefig(f'results_curves/random_beam_search_subplots.png')
plt.show()