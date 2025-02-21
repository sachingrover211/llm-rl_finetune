from world.gym_maze_5x5 import Maze5x5World
from agent.gym_maze_5x5 import Maze5x5Agent
from jinja2 import Environment, FileSystemLoader
import os


def run_training_loop(
    task,
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
):
    assert task == "maze_5x5"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = Maze5x5World(
        gym_env_name, 
        render_mode, 
        max_traj_length,
    )
    agent = Maze5x5Agent(
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
    )

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        agent.train_policy(world, curr_episode_dir)
        # print(f"New Q Table: {agent.q_table}")
        results = agent.evaluate_policy(world, curr_episode_dir)
        print(f"Episode {episode} Evaluation Results: {results}")
