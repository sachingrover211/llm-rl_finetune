import os, json, re
import numpy as np
import pandas as pd
import torch, datasets
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Environment, FileSystemLoader

from agent.finetune.mountain_car_continuous import MountainCarFinetuneAgent
from world.mountain_car import MountainCarContinuousWorld


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
#MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
#MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_POINTS = 4000
RL_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User is looking for a linear control policy "
    "for the continuous Mountain Car Domain. Assistant first thinks about the reasoning process "
    "in the mind and then provides a policy to the user. The reasoning process and the policy "
    "are enclosed within the <think> </think> and <policy> </policy> tags respectively, i.e. "
    "<think> reasoning process here </think><policy> policy here </policy>"
)
LOGDIR = "logs/finetune/qwen2.5_32B_5_epoch"
TEMPLATE_DIR = "agent/policy/templates"
TEMPLATE = "mountaincar_cont_si.j2"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA']="1"
rewards = list()

def create_dataset(world, agent, logdir):
    maximum_possible = 100
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

    jinja2_env = Environment(loader = FileSystemLoader(TEMPLATE_DIR))
    llm_template = jinja2_env.get_template(TEMPLATE)
    prompts = list()
    for _, row in data.iterrows():
        matrix = f"Weights:\n{np.round(row['w0'], decimals = 4)}\n" + \
            f"{np.round(row['w1'], decimals = 4)}\n" + \
            f"Bias:\n{np.round(row['b'], decimals = 4)}"
        prompts.append(llm_template.render({
            "matrix_string": matrix,
            "reward": np.round(row["evaluation"], decimals = 4),
            "optimum": maximum_possible
        }))


    data['prompt'] = prompts
    ds = Dataset.from_pandas(data)

    print(ds)
    train_test = ds.train_test_split(test_size = 0.2)
    #test_val = train_test["test"].train_test_split(test_size=0.5)

    #datasets = DatasetDict({
    #    "train": train_test["train"],
    #    "test": test_val["test"],
    #    "valid": test_val["train"]
    #})

    return train_test["train"], train_test["test"]


def make_rl_conversation(example):
    # print(example, type(example))
    # temp = json.dumps({
    #     "weights": json.dumps([example["w0"], example["w1"], example["b"]]),
    #     "evaluation": example["evaluation"]
    # })
    # weights = json.dumps([example["w0"], example["w1"], example["b"]])
    return {
        "prompt": [
            {"role": "system", "content": RL_SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
    }

def format_soft_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<policy>.*?</policy>$"
    #print("in soft reward", kwargs.keys(), len(completions), completions[0][0].keys())
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    #print(rewards_list)
    return rewards_list

def format_hard_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    """ takes policy structure into account """
    pattern = r"^<think>.*?</think>\s*<policy>[0-9 .-]+[.,\s\]\)\}|]+[0-9 .-]+[.,\s\]\)\}|]+[0-9 .-]+[.,\s\]\)\}|]*</policy>$"
    #print("in hard reward", kwargs.keys(), len(completions), completions[0][0].keys())
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    #print(rewards_list)
    return rewards_list

def evaluate_response(response):
    world = MountainCarContinuousWorld("MountainCarContinuous-v0", None)
    agent = MountainCarFinetuneAgent("logs/fintune", 1, 2, 20, 1000, "", 2000, False, 20)

    reward = 0.0
    policy = agent.parse_response(response[0]['content'])
    if policy is not None and len(policy) == agent.matrix_size[0]:
        agent.initialize_policy(policy)
        reward = agent.rollout_episode_without_logging(world)
        reward = (reward + 200)/300
        reward = np.clip(reward, 0.0, 1.0)

    return reward


def policy_reward(completions, **kwargs):
    # best reward for continuous is around 90 and above, but less than 100. +100 is given if the goal is reached.
    # There can be maximum 999 steps with reward of -0.1 * action^2 (again -100 max possible).
    # So first I add 1000 to ensure it is positive, and then divide by max possible of 200 to normalize it.
    global rewards
    _rewards = [evaluate_response(completion) for completion in completions]
    #print("in policy_rewards", _rewards)
    rewards = _rewards
    return _rewards


def policy_gradient_reward(completions, **kwargs):
    global rewards
    _rewards = list()
    old_rewards = [(r + 200)/300 for r in kwargs["evaluation"]]
    old_rewards = [np.clip(r, 0.0, 1.0) for r in old_rewards]
    #print(f"in policy_gradient_reward, global rewards {rewards}, and earlier evaluations {old_rewards}")
    for ro, rn in zip(old_rewards, rewards):
        if ro != 0:
            _rewards.append((rn-ro)*rn/ro)
        else:
            _rewards.append(rn)

    _rewards = [np.clip(r, 0.0, 1.0) for r in _rewards]
    #print("updated rewards ", _rewards)
    return _rewards

def run():
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
    train_ds.remove_columns(["w0", "w1", "b"])

    print("setting up Finetuning algorithm")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    )

    print("setting up model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    #model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, lora_config)
    #model.to(DEVICE)
    model.print_trainable_parameters()
    training_args = GRPOConfig(
        output_dir=f"{logdir}/mountain_car_continuous-test",
        learning_rate=1e-5,
        remove_unused_columns=False, # to access the solution column in accuracy_reward
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        bf16=True,
        # use_cpu=True,

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
        processing_class = tokenizer,
        reward_funcs=[format_soft_reward, format_hard_reward, policy_reward, policy_gradient_reward],
        args=training_args,
        train_dataset=train_ds,
    )
    print("Training ...")
    trainer.train()
    trainer.save_model(f"{training_args.output_dir}/final")


if __name__ == "__main__":
    run()
