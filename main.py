import yaml
import argparse
from runner import mountain_car_runner


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

    mountain_car_runner.run_training_loop(**config)


if __name__ == "__main__":
    main()
