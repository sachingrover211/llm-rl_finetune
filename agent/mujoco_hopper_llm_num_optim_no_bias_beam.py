from agent.policy.linear_policy_no_bias import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.mujoco_hopper import MujocoHopperWorld
import numpy as np
import re
import heapq
import threading
import os


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

    def rollout_episode(self, world: MujocoHopperWorld, logging_file, record=True):
        state = world.reset()
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
            step_idx += 1
        if logging_file is not None:
            logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        if record:
            self.replay_buffer.add(self.policy.weight, world.get_accu_reward())
        return world.get_accu_reward()

    def rollout_multiple_episodes(
        self, num_episodes, world: MujocoHopperWorld, logging_file, record=True
    ):
        results = []
        for episode in range(num_episodes):
            result = self.rollout_episode(world, logging_file, record)
            results.append(result)
        return results

    def random_warmup(self, world: MujocoHopperWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def generate_new_candidates_random(self, candidate=None):
        if candidate is None:
            candidate = self.policy.weight
        else:
            candidate = candidate[1]
        new_candidates = []
        for i in range(self.num_new_candidate):
            new_candidate = candidate + np.random.normal(0, 0.5, candidate.shape)
            new_candidates.append(new_candidate)
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
            result = self.rollout_multiple_episodes(5, world, None, record=False)
            result = np.mean(result)
            heapq.heappush(self.candidates, (result, self.policy.weight))

        system_prompts = [0] * len(self.candidates)
        new_candidates = [0] * len(self.candidates)
        reasonings_lists = [0] * len(self.candidates)
        threads = []
        for idx, candidate in enumerate(self.candidates):
            thread = threading.Thread(
                target=self.generate_multiple_new_candidates_llm,
                args=(
                    system_prompts,
                    new_candidates,
                    reasonings_lists,
                    idx,
                    world,
                    str_33d_examples(self.replay_buffer),
                    parse_parameters,
                    self.training_episodes,
                    search_std,
                    candidate,
                    temperature,
                ),
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        self.log_down_beam_search(
            logdir, system_prompts, new_candidates, reasonings_lists
        )

        # print(new_candidates)

        new_candidates_2 = []
        for new_candidates_1 in new_candidates:
            new_candidates_2 += new_candidates_1
        new_candidates = new_candidates_2

        # Evaluate the new candidates, and keep the best ones
        for candidate in new_candidates:
            self.policy.update_policy(candidate)
            # result = self.rollout_episode(world, open(f"{logdir}/training_rollout_{self.training_episodes}.txt", "w"), record=False)
            result = self.rollout_multiple_episodes(5, world, None, record=False)
            result = np.mean(result)
            heapq.heappush(self.candidates, (result, candidate))
            if len(self.candidates) > self.beam_width:
                heapq.heappop(self.candidates)

        # Update the policy to the current best weight
        print([x[0] for x in self.candidates])
        # input('Press enter to continue...')
        best_candidate = heapq.nlargest(1, self.candidates)[0]
        print(best_candidate[0])
        best_candidate = best_candidate[1]
        self.policy.update_policy(best_candidate)


        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(20):
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
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
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
