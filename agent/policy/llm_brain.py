import gymnasium as gym
from openai import OpenAI
import random
import numpy as np
import os
import re
import time
from jinja2 import Template
import tiktoken
from utils.llm_interface import get_local_client, query_llm, query_local_llm


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
        llm_ui_template: Template = None,
        model_type = "",
        base_model = "",
    ):
        self.llm_si_template = llm_si_template
        self.llm_ui_template = llm_ui_template
        self.llm_output_conversion_template = llm_output_conversion_template
        #self.client = OpenAI()
        self.llm_conversation = []
        print(llm_model_name)
        #assert llm_model_name in ["o3-mini-2025-01-31", "o3-mini", "o1-preview", "gpt-4o", "gemini-2.0-flash-exp"]
        self.llm_model_name = llm_model_name
        self.is_first_query = True
        self.episode = 0
        # encoder is from open ai API, and we use that to get an estimate of tokens
        self.encoder = tiktoken.encoding_for_model("o1-preview")
        self.tokens = 0
        self.token_limit = 16384
        self.matrix_size = (0, 0)
        self.model_type = model_type
        self.TEXT_KEY = "content"
        self.SYSTEM_ACCOUNT = "system"
        if "gemini" in llm_model_name:
            self.SYSTEM_ACCOUNT = "user"
            self.TEXT_KEY = "parts"
        elif self.model_type in ["HF", "OFFLINE"]:
            self.model, self.tokenizer = get_local_client(llm_model_name, base_model, self.model_type)
        else:
            self.client = OpenAI()


    def create_regex(self):
        single_float = "[0-9 .-]+[, ]+"
        reg = ""
        for _ in range(self.matrix_size[1]):
            reg += single_float

        self.reg = reg[:-5]


    def reset_llm_conversation(self):
        self.llm_conversation = []
        self.tokens = 0


    def add_llm_conversation(self, text, role):
        message = {"role": role}
        message[self.TEXT_KEY] = text
        self.llm_conversation.append(message)
        self.tokens += self.get_token_count(text)


    def remove_llm_conversation(self, index):
        popped_message = self.llm_conversation.pop(index)
        content = popped_message[self.TEXT_KEY]
        self.tokens -= self.get_token_count(content)


    def query_llm(self):
        response = ""
        while self.tokens > self.token_limit:
            # keeping the system prompt and never removing that.
            if len(self.llm_conversation) > 1:
                self.remove_llm_conversation(1)

        if self.model_type in ["HF", "OFFLINE"]:
            response = query_local_llm(self.model, self.tokenizer, self.llm_conversation)
        else:
            response = query_llm(self.client, self.llm_model_name, self.llm_conversation)

        return response


    def delete_model(self):
        if self.model_type == ["HF", "OFFLINE"]:
            remove_local_llm(self.model)


    def parse_q_table(self, q_table_string):
        new_q_values_list = []

        # Update the Q-table based on the new Q-table
        for row in q_table_string.split("\n"):
            if row.strip():
                row = row.split("|")
                #if len(row) == 4:
                if len(row) == 3:
                    try:
                        #position, velocity, action, q_value = row
                        state, action, q_value = row
                        #position = int(position.strip())
                        #velocity = int(velocity.strip())
                        state = int(state.strip())
                        action = int(action.strip())
                        q_value = float(q_value.strip())
                        new_q_values_list.append(
                            ((state,), (action,), q_value)
                        )
                    except:
                        pass

        return new_q_values_list

    def llm_update_q_table(self, q_table, replay_buffer, params = None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render({
            "map": str(params["map"]),
            "replay_buffer_string": str(replay_buffer),
            "q_table_string": str(q_table),
            "average_cost": params["cost"],
            "count": params["count"],
        })

        self.add_llm_conversation(system_prompt, "user")
        new_q_table_with_reasoning = self.query_llm()
        new_q_table_list = self.parse_q_table(new_q_table_with_reasoning)

        trial = 0
        while trial < 3 and len(new_q_table_list) != self.q_dim[0]*self.q_dim[1]:
            self.add_llm_conversation(new_q_table_with_reasoning, "assistant")
            self.add_llm_conversation(
                self.llm_output_conversion_template.render(),
                "user",
            )
            new_q_table = self.query_llm()
            trial += 1

            new_q_table_list = self.parse_q_table(new_q_table)

        # print(f"New Q-table: {new_q_table}")
        return new_q_table_list, new_q_table_with_reasoning


    def llm_update_linear_policy(self, linear_policy, average_reward, replay_buffer = None):
        #self.reset_llm_conversation()
        call_time = time.time()
        repeated_template = self.llm_ui_template if self.llm_ui_template else self.llm_si_template

        if self.is_first_query and repeated_template == self.llm_ui_template:
            system_prompt = self.llm_si_template.render()
            self.add_llm_conversation(system_prompt, self.SYSTEM_ACCOUNT)
            self.is_first_query = False

        if replay_buffer:
            repeated_prompt = repeated_template.render({
                "replay_buffer_string": str(replay_buffer),
                "matrix_string": str(linear_policy),
                "reward": average_reward,
                "episode_number": self.episode
            })
        else:
            repeated_prompt = repeated_template.render({
                "matrix_string": str(linear_policy),
                "reward": average_reward,
                "episode_number": self.episode
            })
        #print(system_prompt)
        #print(repeated_prompt)
        self.add_llm_conversation(repeated_prompt, "user")
        matrix_response_with_reasoning = self.query_llm()

        #print(matrix_response_with_reasoning)
        #if replay_buffer:
        #    # in this case we do not want to track all replay buffers
        #    # thus we will not keep in llm_conversations
        #    self.remove_llm_conversation(-1)
        #    temp = repeated_template.render({
        #        "replay_buffer_string": "",
        #        "matrix_string": str(linear_policy),
        #        "reward": average_reward,
        #        "episode_number": self.episode
        #    })
        #    #print(temp)
        #    self.add_llm_conversation(temp, "user")

        if self.model_type in ["HF", "OFFLINE"]:
            updated_matrix = self.parse_parameters_local(matrix_response_with_reasoning)
        else:
            updated_matrix = self.parse_parameters(matrix_response_with_reasoning)

        # checks if the policy is updated and hasnt parsed wrong parameters
        # from the response.
        if not self.is_policy_updated(linear_policy, updated_matrix, average_reward):
            updated_matrix = []

        trial = 0
        #print("matrix_size", self.matrix_size)
        while trial < 3 and len(updated_matrix) != self.matrix_size[0]:
            print("Could not parse matrix once, trying again", trial, updated_matrix)
            self.add_llm_conversation(
                self.llm_output_conversion_template.render(),
                "user"
            )
            response = self.query_llm()
            if self.model_type in ["HF", "OFFLINE"]:
                updated_matrix = self.parse_parameters_local(response)
            else:
                updated_matrix = self.parse_parameters(response)
            #self.add_llm_conversation(response, "assistant")
            # there is no need to keep the communication
            self.remove_llm_conversation(-1)
            if not self.is_policy_updated(linear_policy, updated_matrix, average_reward):
                updated_matrix = []

            trial += 1

        #print("updated_matrix", updated_matrix)
        self.add_llm_conversation(matrix_response_with_reasoning, "assistant")
        run_time = time.time() - call_time
        return updated_matrix, matrix_response_with_reasoning, run_time


    def parse_parameters(self, parameters_string):
        new_parameters_list = []
        # this is for two dimensions, basically two actions
        # for greater than two actions we will have to re-write this
        # we can't do this individually as some numbers in the conversation
        # will also get parsed.
        # reg = "[0-9 .-]+,[0-9 .-]+"
        # constructed in a different function now

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    temp = re.findall(self.reg, row)
                    if len(temp) == 0:
                        continue
                    for t in temp:
                        parameters_row = [float(x.strip().strip(',')) for x in t.strip().split(",")]
                        new_parameters_list.append(parameters_row)
                    # if len(new_parameters_list) == 5:
                    #    break
                except Exception as e:
                    print(f"Error while parsing {e}")

        return new_parameters_list


    def parse_parameters_local(self, parameters_string):
        # parse params when the response is from local models
        new_parameters_list = list()
        float_continuous = "[0-9 .-]+[.,\s\]\)\}|]+"
        single_float = "[0-9 .-]+"
        new_ps = parameters_string.lower()
        sub_split = ""
        if "weights" in new_ps:
            sub_split = "weights"
        elif "weight matrix" in new_ps:
            sub_split = "weight matrix"

        check_string = ["", new_ps]
        if sub_split != "":
            # the import string is right after the split
            check_string = new_ps.split(sub_split)

        param_size = self.matrix_size[0] * self.matrix_size[1]
        for i in range(1, len(check_string)):
            chk = check_string[i]
            temp = re.findall(float_continuous, chk)
            for t in temp:
                t = re.findall(single_float, t)[0]
                t = t.strip().strip(".").strip()
                if any(ch.isdigit() for ch in t):
                    new_parameters_list.append(float(t))

                if len(new_parameters_list) == param_size:
                    break

            if len(new_parameters_list) == param_size:
                break

        if len(new_parameters_list) == self.matrix_size[0]:
            return np.array(new_parameters_list).reshape(self.matrix_size)

        return new_parameters_list


    def get_token_count(self, message):
        return len(self.encoder.encode(message))

    def is_policy_updated(self, old_policy, new_policy, reward):
        if len(new_policy) != self.matrix_size[0]:
            return True

        is_updated = False

        for i in range(self.matrix_size[0]):
            for j in range(self.matrix_size[1]):
                if str(new_policy[i][j]) in str(reward):
                    return False
                val = old_policy.get_weight_for_matirix(i, j)

                if not val:
                    print("None returned from policy get")

                if new_policy[i][j] != val:
                    is_updated = True

        return is_updated


class LLMBrainStandardized(LLMBrain):
    def __init__(
            self,
            llm_si_template: Template,
            llm_output_conversion_template: Template,
            llm_model_name: str,
            llm_ui_template: Template = None,
            model_type="",
            base_model="",
            env_description = "",
            num_episodes = 400,
    ):
        super().__init__(
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            llm_ui_template,
            model_type,
            base_model,
        )
        self.env_description = env_description
        self.num_episodes = num_episodes

    def str_episode_reward(self, replay_buffer, n):
        all_parameters = []
        for weights, reward in replay_buffer.buffer:
            parameters = weights
            all_parameters.append((parameters.reshape(-1), reward))

        text = ""
        for parameters, reward in all_parameters:
            l = ""
            for i in range(n):
                l += f"params[{i}]: {parameters[i]:.5g}; "
            fxy = reward
            l += f"f(params): {fxy:.2f}\n"
            text += l
        return text

    def llm_update_linear_policy(
            self,
            episode_reward_buffer,
            step_number,
            rank=None,
            optimum=None,
            search_step_size=0.1,
    ):
        call_time = time.time()
        self.rank = rank
        self.reset_llm_conversation()
        text = self.str_episode_reward(episode_reward_buffer, rank)
        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(text),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "env_description": self.env_description,
                "episodes": self.num_episodes
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()
        new_parameters_list = self.parse_parameters_local(new_parameters_with_reasoning)
        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        trial = 0
        while trial < 3 and len(new_parameters_list) != self.rank:
            print("Could not parse matrix once, trying again", trial, new_parameters_list)
            update_query = self.llm_output_conversion_template.render(
                {"rank": rank}
            )
            self.add_llm_conversation(
                update_query,
                "user"
            )
            response = self.query_llm()
            #print(response)
            new_parameters_list = self.parse_parameters_local(response)
            trial += 1

        run_time = time.time() - call_time

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            run_time
        )


    def parse_parameters_local(self, response):
        # This regex looks for integers or floating-point numbers (including optional sign)
        response_array = response.split("\n")
        #print("response:", response_array)
        pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
        results = []

        is_policy_updated = False
        for r in response_array:
            matches = pattern.findall(r)
            # Convert matched strings to float (or int if you prefer to differentiate)
            results = []
            if len(matches) == self.rank:
                for match in matches:
                    results.append(float(match[1]))

                for i, result in enumerate(results):
                    if result != self.policy.get_weight_for_list(i):
                        is_policy_updated = True
                        break

                if is_policy_updated:
                    break

        #print(results)
        return np.array(results).reshape(-1)
