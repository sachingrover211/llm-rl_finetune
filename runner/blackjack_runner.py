from world.blackjack import BlackjackWorld
from agent.blackjack import BlackjackAgent
from agent.blackjack_best import BlackjackAgent as BlackjackBestAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback


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
    warmup_episodes=0,
    warmup_dir=None,
    best_5_q_tables=False,
):
    assert task == "blackjack"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = BlackjackWorld(
        gym_env_name, 
        render_mode,
        max_traj_length,
    )

    if not best_5_q_tables:
        agent = BlackjackAgent(
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
    else:
        agent = BlackjackBestAgent(
            logdir,
            actions,
            states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes
        )

    # for episode in range(num_episodes):
    #     print(f"Episode: {episode}")
    #     # create log dir
    #     curr_episode_dir = f"{logdir}/episode_{episode}"
    #     print(f"Creating log directory: {curr_episode_dir}")
    #     os.makedirs(curr_episode_dir, exist_ok=True)
    #     agent.train_policy(world, curr_episode_dir)
    #     # print(f"New Q Table: {agent.q_table}")
    #     results = agent.evaluate_policy(world, curr_episode_dir)
    #     print(f"Episode {episode} Evaluation Results: {results}")


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
                print(f"{trial_idx + 1}th trial attempt failed with error in training: {e}")
                traceback.print_exc()
                continue

        # print(f"New Q Table: {agent.q_table}")
        results = agent.evaluate_policy(world, curr_episode_dir)
        print(f"Episode {episode} Evaluation Results: {results}")
