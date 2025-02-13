from world.frozen_lake import FrozenLakeWorld
from agent.frozen_lake import FrozenLakeAgent
from jinja2 import Environment, FileSystemLoader
import os
import numpy as np
from utils.result_logger import plot_reward, plot_without_deviation write_to_file


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    actions,
    grid_size,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    record_video,
    use_replay_buffer,
    reset_llm_conversations,
):
    assert task == "grid_world"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = FrozenLakeWorld(
        gym_env_name, 
        render_mode,
        grid_size,
    )
    agent = FrozenLakeAgent(
        logdir,
        actions,
        grid_size,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        record_video,
        use_replay_buffer,
        reset_llm_conversations,
    )
    agent.initialize_policy(grid_size, actions)

    avg = list()
    std = list()
    completed_iterations = list()
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        agent.train_policy(world, curr_episode_dir)
        # print(f"New Q Table: {agent.q_table}")
        result, ci = agent.evaluate_policy(world, curr_episode_dir)
        print(f"Episode {episode} Evaluation Results: {results}")
        print(f"Episode {episode} Completed Rounds: {ci}")
        avg.append(np.average(result))
        std.append(np.std(result))
        completed_iterations.append(ci)

    plot_reward(f"Frozen Lake Grid {grid_size}by{grid_size} average cost", avg, std, logdir, 100)
    write_to_file(logdir, ["Average cost", "Standard deviation", "Completed rounds"], [avg, std, completed_iterations])
    plot_without_deviation(f"Frozen Lake Grid {grid_size}x{grid_size} average completions out of 20")
