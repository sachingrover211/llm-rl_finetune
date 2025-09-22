import os
import numpy as np
from jinja2 import Environment, FileSystemLoader
from utils.result_logger import plot_reward, write_to_file
import matplotlib.pyplot as plt


def training_loop(
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
    step_size,
    reset_llm_conversation,
    print_episode,
    max_limit,
    title,
    world_class,
    agent_class,):
    """Centralized training loop for any environment + agent."""

    # Convert "None" strings to actual None
    render_mode = None if render_mode == "None" else render_mode
    env_desc_file = None if env_desc_file == "None" else env_desc_file

    assert task == task
    # Load Jinja2 templates
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(llm_output_conversion_template_name)
    llm_ui_template = None
    if llm_ui_template_name != "None":
        llm_ui_template = jinja2_env.get_template(llm_ui_template_name)

    root_folder = logdir
    col_titles = ["Average reward", "Standard deviation", "LLM Call Time", "Evaluation Time"]

    for i in range(experiments):
        print(f"################# Experiment Started {i}")
        logdir = f"{root_folder}/experiment_{i}"

        # Initialize world
        world = world_class(gym_env_name, render_mode) if 'gym_env_name' in world_class.__init__.__code__.co_varnames else world_class(render_mode)

        # Initialize agent
        agent = agent_class(
            num_episodes,
            logdir,
            actions,
            states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_ui_template,
            llm_output_conversion_template,
            llm_model_name,
            model_type,
            base_model,
            num_evaluation_episodes,
            warmup_episodes,
            step_size,
            reset_llm_conversation,
            env_desc_file
        )

        agent.initialize_policy(states, actions)

        # Warmup
        curr_episode_dir = f"{logdir}/episode_initial"
        os.makedirs(curr_episode_dir, exist_ok=True)
        agent.add_warmup(world, curr_episode_dir)
        print(f"Initialized weights: {agent.policy}")

        matrix_file = f"{curr_episode_dir}/matrix.txt"
        with open(matrix_file, "w") as f:
            f.write(str(agent.policy))

        # Evaluate initial policy
        results, etime = agent.evaluate_policy(world, curr_episode_dir)

        policies = [str(agent.policy).replace('\n', ', ')]
        avg = [np.average(results)]
        std = [np.std(results)]
        llm_call_times = [0.0]
        eval_times = [etime]
        agent.average_reward = avg[-1]

        # Training loop
        for episode in range(num_episodes):
            print(f"Episode: {episode}")
            curr_episode_dir = f"{logdir}/episode_{episode}"
            os.makedirs(curr_episode_dir, exist_ok=True)

            agent.train_policy(world, curr_episode_dir)
            policies.append(str(agent.policy).replace('\n', ', '))

            results, etime = agent.evaluate_policy(world, curr_episode_dir)
            avg.append(np.average(results))
            std.append(np.std(results))
            llm_call_times.append(agent.run_time)
            eval_times.append(etime)
            agent.average_reward = avg[-1]

            print(f"Episode {episode} Evaluation Results: {avg[-1]}, {std[-1]}")

            if episode > 0 and episode % print_episode == 0:
                _record_results(title, col_titles, [avg, std, llm_call_times, eval_times], logdir, max_limit)

        # Save all policies
        with open(f"{logdir}/policies.txt", "w") as policy_file:
            policy_file.write("\n".join(policies))

        # Final recording
        _record_results(title, col_titles, [avg, std, llm_call_times, eval_times], logdir, max_limit)
        agent.llm_brain.delete_model()
        print(f"################# Experiment End {i}")

def _record_results(graph_title, col_titles, cols, logdir, max_limit=100):
    plot_reward(graph_title, cols[0], cols[1], logdir, max_limit)
    write_to_file(logdir, col_titles, cols)
