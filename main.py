import yaml
import argparse
from runner import (
    mountain_car_runner,
    pendulum_runner,
    blackjack_runner,
    pendulum_continuous_runner,
    mountaincar_continuous_runner,
    swimmer_continuous_runner,
    halfcheetah_continuous_runner,
    mountaincar_continuous_sas_runner,
    swimmer_continuous_sas_runner,
    swimmer_continuous_sas_mean_std_runner,
    swimmer_continuous_random_search_runner,
    swimmer_continuous_llm_num_optim_runner,
    mountaincar_continuous_action_llm_num_optim_runner,
    mountaincar_continuous_action_llm_num_optim_no_bias_runner,
    mountaincar_continuous_action_grid_search_no_bias_runner,
    mujoco_invertedpendulum_llm_num_optim_no_bias_runner,
    mujoco_invertedpendulum_grid_search_no_bias_runner,
    pendulum_grid_search_no_bias_runner,
    pendulum_llm_num_optim_no_bias_runner,
    pendulum_llm_num_optim_no_bias_iter_runner,
    mujoco_hopper_llm_num_optim_no_bias_iter_runner,
    mujoco_hopper_llm_num_optim_no_bias_runner,
    panda_reach_llm_num_optim_no_bias_runner,
    cliff_walking_runner,
    maze3x3_runner,
    maze5x5_runner,
    pong_llm_num_optim_no_bias_runner,
    nav_track_llm_num_optim_no_bias_runner,
    frozen_lake_runner,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["task"] == "mountaincar":
        mountain_car_runner.run_training_loop(**config)
    elif config["task"] == "pendulum":
        pendulum_runner.run_training_loop(**config)
    elif config["task"] == "blackjack":
        blackjack_runner.run_training_loop(**config)
    elif config["task"] == "pendulum_continuous":
        pendulum_continuous_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous":
        mountaincar_continuous_runner.run_training_loop(**config)
    elif config["task"] == "swimmer_continuous":
        swimmer_continuous_runner.run_training_loop(**config)
    elif config["task"] == "halfcheetah_continuous":
        halfcheetah_continuous_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous_sas":
        mountaincar_continuous_sas_runner.run_training_loop(**config)
    elif config["task"] == "swimmer_continuous_sas":
        swimmer_continuous_sas_runner.run_training_loop(**config)
    elif config["task"] == "swimmer_continuous_mean_std_sas":
        swimmer_continuous_sas_mean_std_runner.run_training_loop(**config)
    elif config["task"] == "swimmer_continuous_random_search":
        swimmer_continuous_random_search_runner.run_training_loop(**config)
    elif config["task"] == "swimmer_continuous_llm_num_optim":
        swimmer_continuous_llm_num_optim_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous_action_llm_num_optim":
        mountaincar_continuous_action_llm_num_optim_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous_action_llm_num_optim_no_bias":
        mountaincar_continuous_action_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous_action_grid_search_no_bias":
        mountaincar_continuous_action_grid_search_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "mujoco_invertedpendulum_llm_num_optim_no_bias":
        mujoco_invertedpendulum_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "mujoco_invertedpendulum_grid_search_no_bias":
        mujoco_invertedpendulum_grid_search_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "pendulum_grid_search_no_bias":
        pendulum_grid_search_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "pendulum_llm_num_optim_no_bias":
        pendulum_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "pendulum_llm_num_optim_no_bias_iterative":
        pendulum_llm_num_optim_no_bias_iter_runner.run_training_loop(**config)
    elif config["task"] == "mujoco_hopper_llm_num_optim_no_bias_iterative":
        mujoco_hopper_llm_num_optim_no_bias_iter_runner.run_training_loop(**config)
    elif config["task"] == "mujoco_hopper_llm_num_optim_no_bias":
        mujoco_hopper_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "mountaincar_continuous_action_llm_num_optim_no_bias_norm":
        mountaincar_continuous_action_llm_num_optim_no_bias_runner.run_training_loop_norm(**config)
    elif config["task"] == "panda_reach_llm_num_optim_no_bias":
        panda_reach_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "cliff_walking":
        cliff_walking_runner.run_training_loop(**config)
    elif config["task"] == "maze_3x3":
        maze3x3_runner.run_training_loop(**config)
    elif config["task"] == "maze_5x5":
        maze5x5_runner.run_training_loop(**config)
    elif config["task"] == "pong_llm_num_optim_no_bias":
        pong_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "nav_track_llm_num_optim_no_bias":
        nav_track_llm_num_optim_no_bias_runner.run_training_loop(**config)
    elif config["task"] == "frozen_lake":
        frozen_lake_runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
