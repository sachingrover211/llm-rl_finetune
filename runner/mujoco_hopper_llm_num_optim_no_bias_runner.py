from world.mujoco_hopper import MujocoHopperWorld
from agent.mujoco_hopper_llm_num_optim_no_bias import MujocoHopperLLMNumOptimAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_norm import MujocoHopperLLMNumOptimAgent as NormAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta import MujocoHopperLLMNumOptimAgent as DeltaAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta_descent import MujocoHopperLLMNumOptimAgent as DeltaDescentAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta_descent_reverse import MujocoHopperLLMNumOptimAgent as DeltaDescentReverseAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_beam import MujocoHopperLLMNumOptimBeamAgent as BeamAgent
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
    norm=False,
    delta=False,
    grad_descent=False,
    reverse=False,
    beam=False,
    beam_width=5,
    num_new_candidate=5,
    temperature=1.0,
):
    assert task == "mujoco_hopper_llm_num_optim_no_bias"

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = MujocoHopperWorld(
        gym_env_name, 
        render_mode, 
        max_traj_length,
    )

    if beam:
        agent = BeamAgent(
            logdir,
            dim_actions,
            dim_states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            beam_width=beam_width,
            num_new_candidate=num_new_candidate,
            temperature=temperature,
        )

    else:
        if not norm:
            if not delta:
                agent = MujocoHopperLLMNumOptimAgent(
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
            else:
                if not grad_descent:
                    agent = DeltaAgent(
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
                else:
                    if not reverse:
                        agent = DeltaDescentAgent(
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
                    else:
                        agent = DeltaDescentReverseAgent(
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
        else:
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
        # results = agent.evaluate_policy(world, curr_episode_dir)
        # print(f"Episode {episode} Evaluation Results: {results}")
