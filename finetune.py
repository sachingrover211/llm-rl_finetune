import os, json, re
import numpy as np
import pandas as pd
import torch, datasets
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM

from agent.finetune.mountain_car_continuous import MountainCarFinetuneAgent
from world.mountain_car import MountainCarContinuousWorld

DEVICE = torch.device("cpu")
if torch.cuda.is_available:
    DEVICE = torch.device("cuda")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_POINTS = 2000
RL_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User is looking for a linear control policy "
    "for the continuous Mountain Car Domain. Assistant first thinks about the reasoning process "
    "in the mind and then provides a policy to the user. The reasoning process and the policy "
    "are enclosed within the <think> </think> and <policy> </policy> tags respectively, i.e. "
    "<think> reasoning process here </think><policy> policy here </policy>"
)
LOGDIR = "logs/finetune/qwen_2.5_5_epoch"

def create_dataset(world, agent, logdir):
    os.makedirs(logdir, exist_ok = True)
    file_name = f"{logdir}/mountain_car_dataset.csv"
    data = None
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            data = pd.read_csv(file_name)

    if data is None:
        dataset = list()
        weights = np.random.uniform(-1, 1, (DATA_POINTS, 3)) # weights(2, 1) + bias

        print("Weights initialized ", len(weights), len(weights[0]))
        for weight in weights:
            costs = list()
            agent.initialize_policy(weight)
            costs = agent.evaluate_policy(world, logdir)
            average_cost = np.average(costs)
            row = (weight, average_cost)
            dataset.append(row)

        print("Dataset created")
        data = pd.DataFrame([{"w0": weights[0], "w1": weights[1], "b": weights[2], "evaluation": value} for weights, value in dataset])
        data.to_csv(f'{logdir}/mountain_car_dataset.csv', index = False)

    #mask = np.random.rand(len(data)) < 0.8
    #train = data[mask]
    #test = data[~mask]
    ds = Dataset.from_pandas(data)

    train_test = ds.train_test_split(test_size = 0.2)
    #test_val = train_test["test"].train_test_split(test_size=0.5)

    #datasets = DatasetDict({
    #    "train": train_test["train"],
    #    "test": test_val["test"],
    #    "valid": test_val["train"]
    #})

    return train_test["train"], train_test["test"]


def make_rl_conversation(example):
    #print(example, type(example))
    #temp = json.dumps({
    #    "weights": json.dumps([example["w0"], example["w1"], example["b"]]),
    #    "evaluation": example["evaluation"]
    #})
    weights = json.dumps([example["w0"], example["w1"], example["b"]])
    return {
        "prompt": [
            {"role": "system", "content": RL_SYSTEM_PROMPT},
            {"role": "user", "content": weights},
        ],
    }

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<policy>.*?</policy>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def policy_reward(completions, **kwargs):
    # best reward for continuous is around 90 and above, but less than 100. +100 is given if the goal is reached.
    # There can be maximum 999 steps with reward of -0.1 * action^2 (again -100 max possible).
    # So first I add 1000 to ensure it is positive, and then divide by max possible of 200 to normalize it.
    #print(example, type(example))
    reward = [(_eval + 600)/800 for _eval in kwargs["evaluation"]]
    return reward

def main():
    # create agent and world
    print("CUDA device availability ", torch.cuda.is_available(), DEVICE)
    print("Initialize Agent and World")
    logdir = LOGDIR
    world = MountainCarContinuousWorld("MountainCarContinuous-v0", None)
    agent = MountainCarFinetuneAgent("logs/fintune", 1, 2, 20, 1000, "", 2000, True, 20)

    print("Create Dataset")
    train_ds, test_ds = create_dataset(world, agent, logdir)
    train_ds = train_ds.map(make_rl_conversation)
    test_ds = test_ds.map(make_rl_conversation)
    # train_ds.to(DEVICE)
    # test_ds.to(DEVICE)
    train_ds.remove_columns(["w0", "w1", "b"])

    print("setting up Finetuning algorithm")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    print("setting up model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = get_peft_model(model, lora_config)
    #model.to(DEVICE)
    model.print_trainable_parameters()
    training_args = GRPOConfig(
        output_dir=f"{logdir}/mountain_car_continuous-test",
        learning_rate=1e-5,
        remove_unused_columns=False, # to access the solution column in accuracy_reward
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        bf16=False,

        # Parameters that control de data preprocessing
        max_completion_length=256, # default: 256
        num_generations=4, # default: 8
        max_prompt_length=512, # default: 512

        # Parameters related to reporting and saving
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=True,
        save_strategy="steps",
        save_steps=50,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, policy_reward],
        args=training_args,
        train_dataset=train_ds
    )
    print("Training ...")
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
