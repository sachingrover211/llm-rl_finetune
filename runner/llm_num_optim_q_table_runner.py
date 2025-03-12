from world.mountain_car import MountainCarWorld
from world.gym_maze_3x3 import Maze3x3World
from world.nim import NimWorld
from agent.llm_num_optim_q_table import LLMNumOptimQTableAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np
import itertools


def run_training_loop(
    task,
    world,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    actions,
    states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    num_training_rollouts,
    warmup_episodes,
    warmup_dir,
    optimum=1000,
):
    assert task in ["llm_num_optim_q_table"]

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    if world == "mountaincar":
        world = MountainCarWorld(
            gym_env_name,
            render_mode,
            num_position_bins=10,
            num_velocity_bins=10,
            max_traj_length=max_traj_length,
            tile_state=False,
        )
    elif world == "gym_maze_3x3":
        world = Maze3x3World(
            gym_env_name,
            render_mode,
            max_traj_length,
        )
    elif world == "nim":
        world = NimWorld(
            gym_env_name,
            render_mode,
            10,
            10,
        )

    agent = LLMNumOptimQTableAgent(
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        num_training_rollouts,
        optimum,
    )

    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        agent.random_warmup(world, warmup_dir, warmup_episodes)
    else:
        agent.replay_buffer.load(warmup_dir)
    
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        for trial_idx in range(5):
            try:
                agent.train_policy(world, curr_episode_dir)
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(
                    f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                )
                traceback.print_exc()
                continue
        # results = agent.evaluate_policy(world, curr_episode_dir)
        # print(f"Episode {episode} Evaluation Results: {results}")
