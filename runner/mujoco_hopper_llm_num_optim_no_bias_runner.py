from world.mujoco_hopper import MujocoHopperWorld
from agent.mujoco_hopper_llm_num_optim_no_bias import MujocoHopperLLMNumOptimAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_norm import MujocoHopperLLMNumOptimAgent as NormAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta import MujocoHopperLLMNumOptimAgent as DeltaAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta_descent import MujocoHopperLLMNumOptimAgent as DeltaDescentAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_delta_descent_reverse import MujocoHopperLLMNumOptimAgent as DeltaDescentReverseAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_beam import MujocoHopperLLMNumOptimBeamAgent as BeamAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_beam_random import MujocoHopperLLMNumOptimBeamAgent as RandomBeamAgent
from agent.mujoco_hopper_llm_num_optim_no_bias_beam_random_llm_reward import MujocoHopperLLMNumOptimBeamAgent as LLMRewardBeamAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np


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
    random=False,
    beam_width=5,
    num_new_candidate=5,
    temperature=1.0,
    using_llm=False,
    forward_reward_only=False,
    llm_reward=False,
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

    if llm_reward:
        agent = LLMRewardBeamAgent(
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
        if beam:
            if not random:
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
                agent = RandomBeamAgent(
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
                    using_llm=using_llm,
                    forward_reward_only=forward_reward_only,
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
    agent.policy.weight = np.array([[1.2179141159791909, 1.1413004009355927, 1.530527164426801], [1.934992011923374, -1.992556031887027, -5.6750327954581445], [-0.9889273156146797, 2.725096407998663, 1.0076919583463848], [-3.3373615342241, 0.5493752410047001, -0.6974548435477655], [-0.9632137541897214, 0.8489260295622312, -4.832464743147694], [-1.424289095444288, 3.413227964345549, -6.40001919117302], [-3.7219523837078836, 2.654367257825696, -0.5655489863827761], [0.18369421570808664, -2.7739190734950787, 0.5843377351174016], [-2.6820011043229792, 1.1744371110287823, 2.094140622202922], [-2.2818078767753085, -1.0850135276942119, 2.3793466853003604], [-1.1098357147595808, -0.011416843977954817, 0.06097992498272066]])
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
                traceback.print_exc()
                continue
        # results = agent.evaluate_policy(world, curr_episode_dir)
        # print(f"Episode {episode} Evaluation Results: {results}")
