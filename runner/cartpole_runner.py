from world.cartpole import CartpoleWorld
from agent.cartpole import CartpoleAgent, ContinuousCartpoleAgent
from jinja2 import Environment, FileSystemLoader
import os
import numpy as np


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    continuous,
    num_position_bins,
    num_theta_bins,
    logdir,
    actions,
    states,
    max_traj_count,
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
    assert task == "cartpole"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    llm_ui_template = None
    if llm_ui_template != "None":
        llm_ui_template = jinja2_env.get_template(llm_ui_template_name)

    world = CartpoleWorld(
        render_mode,
        continuous,
        num_position_bins,
        num_theta_bins,
    )

    if continuous:
        agent = ContinuousCartpoleAgent(
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
    else:
        agent = CartpoleAgent(
            logdir,
            actions,
            states,
            num_position_bins,
            num_theta_bins,
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
    os.makedirs(curr_episode_dir, exist_ok=True)
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
        # print(f"New Q Table: {agent.q_table}")
        results = agent.evaluate_policy(world, curr_episode_dir)
        avg.append(np.average(results))
        std.append(np.std(results))
        agent.average_reward = avg[-1]
        print(results)
        print(f"Episode {episode} Evaluation Results: {avg[-1]}, {std[-1]}")

    print("Average", avg)
    print("Standard deviation", std)
