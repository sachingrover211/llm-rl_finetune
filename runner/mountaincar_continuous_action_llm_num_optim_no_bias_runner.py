from world.mountaincar_continuous_action import MountaincarContinuousActionWorld
from agent.mountain_car_continuous_llm_num_optim_no_bias import MountaincarContinuousActionLLMNumOptimAgent
from agent.mountain_car_continuous_llm_num_optim_no_bias_norm import MountaincarContinuousActionLLMNumOptimAgent as NormAgent
from jinja2 import Environment, FileSystemLoader
import os


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    search_std,
):
    assert task == "mountaincar_continuous_action_llm_num_optim_no_bias"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = MountaincarContinuousActionWorld(
        gym_env_name, 
        render_mode, 
        max_traj_length,
    )
    agent = MountaincarContinuousActionLLMNumOptimAgent(
        logdir,
        dim_actions,
        dim_states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
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
                agent.train_policy(world, curr_episode_dir, search_std)
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(f"{trial_idx + 1}th trial attempt failed with error in training: {e}")
                continue
        results = agent.evaluate_policy(world, curr_episode_dir)
        print(f"Episode {episode} Evaluation Results: {results}")



def run_training_loop_norm(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    search_std,
):
    assert task == "mountaincar_continuous_action_llm_num_optim_no_bias_norm"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = MountaincarContinuousActionWorld(
        gym_env_name, 
        render_mode, 
        max_traj_length,
    )
    agent = NormAgent(
        logdir,
        dim_actions,
        dim_states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
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
                agent.train_policy(world, curr_episode_dir, search_std)
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(f"{trial_idx + 1}th trial attempt failed with error in training: {e}")
                continue
        results = agent.evaluate_policy(world, curr_episode_dir)
        print(f"Episode {episode} Evaluation Results: {results}")
