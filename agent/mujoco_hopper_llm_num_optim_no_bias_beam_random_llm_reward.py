from agent.policy.linear_policy_no_bias import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.mujoco_hopper import MujocoHopperWorld
import numpy as np
import re
import heapq
import threading
import os
from openai import OpenAI
import traceback


class MujocoHopperLLMNumOptimBeamAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        beam_width=5,
        num_new_candidate=5,
        temperature=1.0,
        using_llm=False,
        forward_reward_only=False,
    ):
        self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.candidates = []
        self.beam_width = beam_width
        self.num_new_candidate = num_new_candidate
        heapq.heapify(self.candidates)
        self.temperature = temperature
        self.using_llm = using_llm
        self.forward_reward_only = forward_reward_only

    def rollout_episode(
        self, world: MujocoHopperWorld, logging_file, record=True, new_reward=True
    ):
        
        all_states = []
        state = world.reset(new_reward=new_reward)
        state = np.expand_dims(state, axis=0)
        if logging_file is not None:
            logging_file.write(
                f"{', '.join([str(x) for x in self.policy.weight.reshape(-1)])}\n"
            )
            logging_file.write(f"parameter ends\n\n")
            logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, 3))
            next_state, reward, done = world.step(action)
            if logging_file is not None:
                logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            state = next_state
            all_states.append(state)
            step_idx += 1
        if logging_file is not None:
            logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        if record:
            self.replay_buffer.add(self.policy.weight, world.get_accu_reward())
        return np.array(all_states), world.get_accu_reward()

    def rollout_multiple_episodes(
        self, num_episodes, world: MujocoHopperWorld, logging_file, record=True
    ):
        all_states, all_rewards = [], []
        for episode in range(num_episodes):
            states, reward = self.rollout_episode(
                world, logging_file, record, new_reward=self.forward_reward_only
            )
            all_states.append(states)
            all_rewards.append(reward)
        return all_states, all_rewards

    def random_warmup(self, world: MujocoHopperWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(
                world, logging_file, new_reward=self.forward_reward_only
            )
            print(f"Result: {result}")

    def generate_random_numbers(self, mean, std, shape=(11, 3), bulk=1):

        count = np.prod(shape)

        def parse_numbers(response_text):
            numbers = []
            for line in response_text.split("\n"):
                try:
                    number = float(line)
                    numbers.append(number)
                except ValueError:
                    print(f"Failed to parse number: {line}")
            # print(numbers)
            numbers = numbers[:count]
            return numbers

        prompt = (
            f"Generate {count + 5} random numbers from the normal distribution of the mean of 0 and the standard deviation of 10. "
            "Provide one number in each line. Do no include any other text or characters."
        )

        for idx in range(5):
            try:
                response = self.llm_brain.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    n=bulk,
                )

                # Extract and parse response
                if bulk == 1:
                    response_text = response.choices[0].message.content.strip()

                    numbers = parse_numbers(response_text)
                    if len(numbers) != count:
                        raise ValueError(
                            f"Expected {count} numbers, but got {len(numbers)}"
                        )
                    numbers = (np.array(numbers).reshape(shape) / 10 + mean) * std
                    return numbers
                else:
                    all_numbers = []
                    for i in range(bulk):
                        response_text = response.choices[i].message.content.strip()
                        numbers = parse_numbers(response_text)
                        if len(numbers) != count:
                            raise ValueError(
                                f"Expected {count} numbers, but got {len(numbers)}"
                            )
                        numbers = (np.array(numbers).reshape(shape) / 10 + mean) * std
                        all_numbers.append(numbers)
                    return np.array(all_numbers)
            except Exception as e:
                print(f"{idx}th trial failed with error: {e}")
                traceback.print_exc()

    def generate_new_candidates_random(
        self, candidate=None, using_llm=False, new_candidates_ret=None, idx=0
    ):
        if candidate is None:
            candidate = self.policy.weight
        else:
            candidate = candidate[1]
        new_candidates = []
        if not using_llm:
            for i in range(self.num_new_candidate):
                new_candidate = candidate + np.random.normal(0.0, 0.5, candidate.shape)
                new_candidates.append(new_candidate)
        else:
            delta_new_candidate = self.generate_random_numbers(
                0, 0.5, candidate.shape, self.num_new_candidate
            )
            with open(os.path.join(self.logdir, "delta_new_candidate.txt"), "a") as f:
                for i in range(self.num_new_candidate):
                    new_candidate = candidate + delta_new_candidate[i]
                    f.write(
                        f"{[x for x in list(delta_new_candidate[i].reshape(-1))]}\n\n"
                    )
                    new_candidates.append(new_candidate)

        new_candidates_ret[idx] = new_candidates
        return new_candidates

    def generate_new_candidates_llm(
        self,
        world,
        replay_buffer_str,
        parse_parameters,
        step_number,
        search_std,
        candidate=None,
    ):

        def candidate_to_str(candidate):
            reward, params = candidate
            params = np.array(params).reshape(-1)
            l = ""
            for i in range(33):
                l += f"params[{i}]: {params[i]:.15g}; "
            l += f"f(params): {reward:.2f}\n"
            return l

        if candidate is None:
            reward = self.rollout_multiple_episodes(5, world, None, record=False)
            reward = np.mean(reward)
            candidate = (reward, self.policy.weight)
        new_candidates = []
        for i in range(self.num_new_candidate):
            candidate_str = candidate_to_str(candidate)
            new_parameters, reasoning_str = (
                self.llm_brain.llm_propose_parameters_num_optim_based_on_anchor(
                    replay_buffer_str,
                    parse_parameters,
                    step_number,
                    search_std,
                    candidate_str,
                )
            )
            new_candidates.append(new_parameters)
        return new_candidates

    def generate_multiple_new_candidates_llm(
        self,
        system_prompt_return,
        new_candidates_return,
        reasonings_list_return,
        idx,
        world,
        replay_buffer_str,
        parse_parameters,
        step_number,
        search_std,
        candidate=None,
        temperature=1.0,
    ):

        def candidate_to_str(candidate):
            reward, params = candidate
            params = np.array(params).reshape(-1)
            l = ""
            for i in range(33):
                l += f"params[{i}]: {params[i]:.15g}; "
            l += f"f(params): {reward:.2f}\n"
            return l

        if candidate is None:
            reward = self.rollout_multiple_episodes(5, world, None, record=False)
            reward = np.mean(reward)
            candidate = (reward, self.policy.weight)

        candidate_str = candidate_to_str(candidate)

        system_prompt, new_parameters_list, reasonings_list = (
            self.llm_brain.llm_propose_multiple_parameters_num_optim_based_on_anchor(
                replay_buffer_str,
                parse_parameters,
                step_number,
                search_std,
                candidate_str,
                self.num_new_candidate,
                temperature,
            )
        )

        system_prompt_return[idx] = system_prompt
        new_candidates_return[idx] = new_parameters_list
        reasonings_list_return[idx] = reasonings_list

    def generate_new_candidates_llm_thread(
        self,
        new_candidates_all,
        new_idx,
        world,
        replay_buffer_str,
        parse_parameters,
        step_number,
        search_std,
        candidate=None,
    ):

        def candidate_to_str(candidate):
            reward, params = candidate
            params = np.array(params).reshape(-1)
            l = ""
            for i in range(33):
                l += f"params[{i}]: {params[i]:.15g}; "
            l += f"f(params): {reward:.2f}\n"
            return l

        if candidate is None:
            reward = self.rollout_multiple_episodes(5, world, None, record=False)
            reward = np.mean(reward)
            candidate = (reward, self.policy.weight)
        new_candidates = [0] * self.num_new_candidate
        candidate_str = candidate_to_str(candidate)
        threads = []
        for i in range(self.num_new_candidate):
            thread = threading.Thread(
                target=self.llm_brain.llm_propose_parameters_num_optim_based_on_anchor_thread,
                args=(
                    new_candidates,
                    i,
                    replay_buffer_str,
                    parse_parameters,
                    step_number,
                    search_std,
                    candidate_str,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        new_candidates_all[new_idx] = new_candidates
        return new_candidates

    def log_down_beam_search(
        self, logdir, system_prompts, new_candidates, reasonings_lists
    ):

        for idx in range(len(system_prompts)):
            with open(f"{logdir}/beam_search_candidate_{idx}.txt", "w") as f:
                f.write(f"System prompt: \n{system_prompts[idx]}\n\n\n")

                for i in range(len(new_candidates[idx])):
                    f.write(f"Reasoning {i}: \n{reasonings_lists[idx][i]}\n\n")
                    f.write(f"New candidate {i}:\n")
                    f.write(
                        f"{[str(x) for x in new_candidates[idx][i].reshape(-1)]}\n\n\n"
                    )
                f.close()

    def llm_find_best_candidates(self, logdir, trajs, N):
        # trajs: [(traj, weight), ...]
        N = min(N, len(trajs))

        def format_trajectory(traj):
            # Format the trajectory
            traj_str = ""
            for i in range(0, traj.shape[0], 5):
                traj_str += f"Step {i+1}:\n"
                for j in range(traj.shape[1]):
                    traj_str += f"{traj[i, j]:.2f} "
                    if (j + 1) % 11 == 0:
                        traj_str += "\n"
                traj_str += "\n"

            return traj_str

        def format_trajectories(trajs):
            trajs_str = ""
            for idx, (traj, weight, reward) in enumerate(trajs):
                trajs_str += f"Trajectory {idx}:\n"
                trajs_str += format_trajectory(traj)
                trajs_str += f"\n\n"
            return trajs_str

        def build_prompt(traj_str):
            # Write the prompt
            prompt = f"""
### You will see the trajectory of an execution of mujoco hopper environment. Your goal is to help me find out the best {N} trajectories out of the following {len(trajs)} trajectories.

Each trajectory is a list of states of 11 dimensions. The 11 dimensions of the state are:
1. z-position of the torso (vertical height)
2. Orientation (the pitch angle around the y-axis)
3. Thigh joint angle
4. Leg joint angle
5. Foot joint angle
6. x-velocity (forward speed)
7. z-velocity (vertical speed)
8. Orientation (pitch) velocity
9. Thigh joint velocity
10. Leg joint velocity
11. Foot joint velocity

### The goal of hopper is to reach the x as forward as possible. The hopper is also subject to gravity and other forces.


### Your goal is to analyze the trajectories and pick out the best {N} trajectories. Rank them from best to Nth best.
Please follow the exact output format without any changes: Firstly, create the analysis for each of the trajectories. Here, you should write 1 line for each trajectory, so there needs to be {len(trajs)} lines. Then, write the conclusion with the best {N} trajectories, where you should list {N} integer numbers. The exact format should be exactly same as the following example:

(In this example, there are 4 trajectories and the best 2 trajectories are 2 and 1. Your actual question will have different number of trajectories and different best trajectories.)
# Analysis:
Trajectory 0: <Description of the trajectory>
Trajectory 1: <Description of the trajectory>
Trajectory 2: <Description of the trajectory>
Trajectory 3: <Description of the trajectory>

# Conlusion:
The best 2 trajectories are:
Trajectory 2, Trajectory 1

### Next, you will see {len(trajs)} trajectories. Please analyze the trajectories and pick out the best {N} trajectories. Rank them from best to Nth best. Give me your results according to the example format.
{traj_str}
            """

            return prompt

        def ask_llm(prompt):
            # client = OpenAI(
            #     base_url="http://en4146091l.scai.dhcp.asu.edu:11434/v1",
            #     api_key="ollama",  # required, but unused
            # )

            client = OpenAI()

            completion = client.chat.completions.create(
                model="gpt-4o",
                # model="deepseek-r1:70b",
                messages=[
                    {"role": "developer", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            result = completion.choices[0].message.content
            print('llm result: ', result)
            return result

        def parse_result(result):
            # Parse the result
            result = result.split('\n')[-1]
            result = result.strip()
            result = result.split(':')[-1].strip()
            result = result.split(", ")
            result = [int(x.strip().split(" ")[1]) for x in result]
            return result

        # Format the trajectories
        trajs_str = format_trajectories(trajs)
        # Build the prompt
        prompt = build_prompt(trajs_str)
        # Ask the LLM
        result = ask_llm(prompt)
        # Parse the result
        result_parsed = parse_result(result)

        if logdir:
            with open(f"{logdir}/llm_reward.txt", "w") as f:
                f.write(f"Prompt: \n{prompt}\n\n")
                f.write(f"Result: \n{result}\n\n")
                f.write(f"Result parsed: \n{result_parsed}\n\n")
                f.write(f"Picked candidates: \n{[trajs[i][2] for i in result_parsed]}\nfrom {[x[2] for x in trajs]}\n")
                f.close()
        # Return the best trajectories
        return [trajs[i] for i in result_parsed]

    def train_policy(self, world: MujocoHopperWorld, logdir, search_std):

        def parse_parameters(input_text):
            # This regex looks for integers or floating-point numbers (including optional sign)
            s = input_text.split("\n")[0]
            # print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)

            # Convert matched strings to float (or int if you prefer to differentiate)
            results = []
            for match in matches:
                results.append(float(match[1]))
            # print(results)
            assert len(results) == 33
            return np.array(results).reshape((11, 3))

        def str_33d_examples(replay_buffer: EpisodeRewardBufferNoBias):

            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(33):
                    l += f"params[{i}]: {parameters[i]:.15g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text

        temperature = self.temperature

        # Generate new candidates for each candidate
        if len(self.candidates) == 0:
            traj, result = self.rollout_multiple_episodes(1, world, None, record=False)
            traj = traj[0]
            result = result[0]
            self.candidates.append([traj, self.policy.weight, result])

        system_prompts = [0] * len(self.candidates)
        new_candidates = [0] * len(self.candidates)
        reasonings_lists = [0] * len(self.candidates)

        threads = []
        for idx, candidate in enumerate(self.candidates):
            thread = threading.Thread(
                target=self.generate_new_candidates_random,
                args=(candidate, self.using_llm, new_candidates, idx),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # for idx, candidate in enumerate(self.candidates):
        #     new_candidates[idx] = self.generate_new_candidates_random(
        #         candidate, using_llm=self.using_llm
        #     )

        # self.log_down_beam_search(
        #     logdir, system_prompts, new_candidates, reasonings_lists
        # )

        # print(new_candidates)

        new_candidates_2 = []
        for new_candidates_1 in new_candidates:
            new_candidates_2 += new_candidates_1
        new_candidates = new_candidates_2

        # Evaluate the new candidates, and keep the best ones
        trajs = []
        for candidate in new_candidates:
            self.policy.update_policy(candidate)
            traj, result = self.rollout_multiple_episodes(1, world, None, record=False)
            traj = traj[0]
            result = result[0]
            trajs.append((traj, candidate, result))

        self.candidates = self.llm_find_best_candidates(logdir, trajs, self.beam_width)

        # Update the policy to the current best weight
        print([x[2] for x in self.candidates])
        # input('Press enter to continue...')
        best_candidate = self.candidates[0]
        print(best_candidate[2])
        best_candidate = best_candidate[1]
        self.policy.update_policy(best_candidate)

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(20):
            result = self.rollout_episode(
                world,
                logging_file,
                record=False,
                new_reward=False,
            )
            results.append(result[1])
        print(f"Results: {results}")
        result = np.mean(results)
        print(f"Mean Results: {result}")
        self.replay_buffer.add(self.policy.weight, result)

        self.training_episodes += 1

    def evaluate_policy(self, world: MujocoHopperWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(
                world, logging_file, record=False, new_reward=False
            )
            results.append(result)
        return results
