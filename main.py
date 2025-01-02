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
    else:
        raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
