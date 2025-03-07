import yaml
import argparse
from runner import mountain_car_runner, pendulum_runner, blackjack_runner, cartpole_runner, frozen_lake_runner, hopper_runner, mountain_car_cont_runner


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
    elif config["task"] == "cartpole":
        cartpole_runner.run_training_loop(**config)
    elif config["task"] == "grid_world":
        frozen_lake_runner.run_training_loop(**config)
    elif config["task"] == "hopper":
        hopper_runner.run_training_loop(**config)
    elif config["task"] == "mc_continuous":
        mountain_car_cont_runner.run_training_loop(**config)
    else:
        msg = f"Task {config['task']} not recognized."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
