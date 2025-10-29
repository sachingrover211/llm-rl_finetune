from world.frozen_lake import FrozenLakeWorld
from agent.frozen_lake import FrozenLakeAgent
from jinja2 import Environment, FileSystemLoader
import os
import numpy as np
from utils.result_logger import plot_reward, plot_without_deviation, write_to_file


def run_training_loop(
    task,
    experiments,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    actions,
    states,
    max_traj_count,
    max_traj_length,
    warmup_episodes,
    template_dir,
    llm_si_template_name,
    llm_ui_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    env_desc_file,
    model_type,
    base_model,
    num_evaluation_episodes,
    record_video,
    use_replay_buffer,
    step_size,
    reset_llm_conversation,
    print_episode,
    max_limit,
    title,
):
    assert task == "grid_world"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    num_states = len(states[0]) if states and len(states) > 0 else 16
    grid_size = int(np.sqrt(num_states))

    root_folder = logdir
    max_limit = 100
    for i in range(experiments):
        print(f"################# Experiment Started {i}")
        logdir = f"{root_folder}/experiment_{i}"
        world = FrozenLakeWorld(
            render_mode,
            grid_size,
        )
        os.makedirs(logdir, exist_ok=True)
        # save image of the grid world
        world.save_domain(logdir, i)
        agent = FrozenLakeAgent(
            num_episodes=num_episodes,
            logdir=logdir,
            actions=actions,
            states=states,
            max_traj_count=max_traj_count,
            max_traj_length=max_traj_length,
            llm_si_template=llm_si_template,
            llm_ui_template=llm_ui_template,
            llm_output_conversion_template=llm_output_conversion_template,
            llm_model_name=llm_model_name,
            model_type=model_type,
            base_model=base_model,
            num_evaluation_episodes=num_evaluation_episodes,
            warmup_episodes=warmup_episodes,
            step_size=step_size,
            reset_llm_conversations=reset_llm_conversation,
            env_desc_file=env_desc_file,
            record_video=record_video,
            use_replay_buffer=use_replay_buffer,
        )
        agent.initialize_policy(world, grid_size, actions)
        curr_episode_dir = f"{logdir}/episode_initial"
        os.makedirs(curr_episode_dir, exist_ok=True)
        result, ci = agent.evaluate_policy(world, curr_episode_dir)

        avg = list()
        std = list()
        completed_iterations = list()
        avg_cost = np.average(result)
        avg.append(avg_cost)
        std.append(np.std(result))
        completed_iterations.append(ci)
        for episode in range(num_episodes):
            print(f"Episode: {episode}")
            # create log dir
            curr_episode_dir = f"{logdir}/episode_{episode}"
            print(f"Creating log directory: {curr_episode_dir}")
            os.makedirs(curr_episode_dir, exist_ok=True)
            agent.train_policy(world, curr_episode_dir, avg_cost, ci)
            # print(f"New Q Table: {agent.q_table}")
            result, ci = agent.evaluate_policy(world, curr_episode_dir)
            print(f"Episode {episode} Evaluation Results: {result}")
            print(f"Episode {episode} Completed Rounds: {ci}")
            avg_cost = np.average(result)
            avg.append(avg_cost)
            std.append(np.std(result))
            completed_iterations.append(ci)

        plot_reward(f"Frozen Lake Grid {grid_size}by{grid_size} average cost", avg, std, logdir, 100)
        write_to_file(logdir, ["Average cost", "Standard deviation", "Completed rounds"], [avg, std, completed_iterations])
        plot_without_deviation(f"Frozen Lake Grid {grid_size}x{grid_size} completions out of 20", "# of Completions", completed_iterations, logdir, 20.0)
