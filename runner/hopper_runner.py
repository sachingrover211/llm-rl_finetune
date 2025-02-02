from world.hopper import HopperWorld
from agent.hopper import HopperAgent
from jinja2 import Environment, FileSystemLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.result_logger import plot_reward, write_to_file


def run_training_loop(
    task,
    experiments,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    actions,
    states,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_ui_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    record_video,
    use_replay_buffer,
    reset_llm_conversation,
):
    assert task == "hopper"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    llm_ui_template = None
    if llm_ui_template_name != "None":
        llm_ui_template = jinja2_env.get_template(llm_ui_template_name)

    root_folder = logdir
    for i in range(experiments):
        print(f"################# Experiment Started {i}")
        logdir = f"{root_folder}/experiment_{i}"

        world = HopperWorld(
            render_mode,
        )

        agent = HopperAgent(
            logdir,
            actions,
            states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_ui_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            record_video,
            use_replay_buffer,
            reset_llm_conversation
        )

        agent.initialize_policy(states, actions)
        curr_episode_dir = f"{logdir}/episode_initial"
        print(f"Initialized weights: {str(agent.policy)}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        matrix_file = f"{curr_episode_dir}/matrix.txt"
        with open(matrix_file, "w") as f:
            f.write(str(agent.policy))

        results = agent.evaluate_policy(world, curr_episode_dir)

        avg = list()
        std = list()
        avg.append(np.average(results))
        std.append(np.std(results))
        agent.average_reward = avg[-1]
        for episode in range(num_episodes):
            print(f"Episode: {episode}")
            # create log dir
            curr_episode_dir = f"{logdir}/episode_{episode}"
            print(f"Creating log directory: {curr_episode_dir}")
            os.makedirs(curr_episode_dir, exist_ok=True)
            agent.train_policy(world, curr_episode_dir)
            print(f"New Matrix: {str(agent.policy)}")
            results = agent.evaluate_policy(world, curr_episode_dir)
            avg.append(np.average(results))
            std.append(np.std(results))
            agent.average_reward = avg[-1]
            print(results)
            print(f"Episode {episode} Evaluation Results: {avg[-1]}, {std[-1]}")
            if episode > 0 and episode % print_episode == 0:
                record_results(avg, std, logdir, max_limit)

        print("Average", avg)
        print("Standard deviation", std)
        print(f"################# Experiment End {i}")

def record_results(avg, std, logdir, max_limit = 2200):
        plot_reward("Hopper Trajectory No Context", avg, std, logdir, max_limit)
        write_to_file(logdir, ["Average reward", "Standard deviation"], [avg, std])
