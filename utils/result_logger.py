import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

is_first = True

def plot_reward(title, reward, sd, logdir, max_limit, index = 0, do_clear_plot = True):
    episodes = list(range(len(reward)))
    global is_first

    if do_clear_plot or is_first:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.ylim(0, max_limit*1.01)
        plt.title(title)
        plt.grid(True)
    plt.plot(episodes, reward, label=f'Experiment {index}')
    plt.fill_between(episodes, np.array(reward) - np.array(sd), np.array(reward) + np.array(sd), alpha=0.2)
    plt.legend(loc="lower right")
    if do_clear_plot:
        plt.savefig(f"{logdir}/plot{index}.png")
        plt.clf()

    is_first = False

def plot_without_deviation(title, ylabel, avg, logdir, max_limit, index = 0, do_clear_plot = True):
    episodes = list(range(len(reward)))
    global is_first

    if do_clear_plot or is_first:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.ylim(0, max_limit*1.01)
        plt.title(title)
        plt.grid(True)

    plt.plot(episodes, avg, label=f"Experiment {index}")
    plt.legent(loc="lower right")

    if do_clear_plot:
        plt.savefix(f"{logdir}/plot_{title}_{index}.png")
        plt.clf()

    is_first = False


def plot_traces_from_csv(file_names, logdir, max_limit, is_new_plots = True):
    title = str(logdir.split("/")[-1])
    for ix, name in enumerate(file_names):
        print(name)
        df = pd.read_csv(name)
        #if df["Average reward"].iloc[-1] > 450.0:
        plot_reward(title, df["Average reward"], df["Standard deviation"], logdir, max_limit, ix, is_new_plots)

    plt.savefig(f"{logdir}/plot.png")
    plt.close()


def get_average_reward(file_names, logdir, max_limits, index = 0):
    df = pd.DataFrame()
    for ix, name in enumerate(file_names):
        data = pd.read_csv(name)
        #if data['Average reward'].iloc[-1] > 450.0:
        df[f'avg_reward_{ix}'] = data['Average reward']

    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    title = f"Average reward over {ix + 1} experiments"
    plot_reward(title, df['mean'], df['std'], logdir, max_limits, index)
    print("Last Episode average reward -- ", df['mean'].iloc[-1])


def get_grouped_average_reward(file_name, _size, logdir, max_limits):
    ix = 0
    index = 0
    while ix < len(file_name):
        get_average_reward(file_name[ix:ix+_size], logdir, max_limits, index)
        ix += _size
        index += 1


def write_to_file(logdir, header, values):
    temp = dict()
    for _head, col in zip(header, values):
        temp[_head] = col

    df = pd.DataFrame(temp)
    df.to_csv(f"{logdir}/results.csv", index=False, header=True, encoding='utf-8')


def plot_hist(files, index):
    values = list()
    for name in files:
        df = pd.read_csv(name)
        values.append(df['Average reward'].iloc[index])

    print(values)
    plt.hist(values)
    plt.savefig("histogram.png")


if __name__ == "__main__":
    # running code for making all traces in one plot
    names = [
        "../logs/gemini/cartpole_0_100/experiment_0/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_1/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_2/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_3/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_4/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_5/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_6/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_7/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_8/results.csv",
        "../logs/gemini/cartpole_0_100/experiment_9/results.csv",
    ]
    logdir = "../logs/gemini/cartpole_0_100"
    plot_traces_from_csv(names, logdir, 500.0, False)
    #get_average_reward(names, logdir, 500.0)
    #plot_hist(names, -1)
