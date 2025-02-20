import gymnasium as gym
from openai import OpenAI
import random
import numpy as np
import os
import re
import time
from jinja2 import Template
import tiktoken
from utils.llm_interface import get_client, query_llm


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
        llm_ui_template: Template = None,
    ):
        self.llm_si_template = llm_si_template
        self.llm_ui_template = llm_ui_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.client = OpenAI()
        self.llm_conversation = []
        print(llm_model_name)
        assert llm_model_name in ["o3-mini-2025-01-31", "o3-mini", "o1-preview", "gpt-4o", "gemini-2.0-flash-exp"]
        self.llm_model_name = llm_model_name
        self.is_first_query = True
        self.episode = 0
        # encoder is from open ai API, and we use that to get an estimate of tokens
        self.encoder = tiktoken.encoding_for_model("o1-preview")
        self.tokens = 0
        self.token_limit = 500000
        self.matrix_size = 0
        self.model_type = "openai"
        self.TEXT_KEY = "content"
        self.SYSTEM_ACCOUNT = "system"
        if "gemini" in llm_model_name:
            self.model_type = "gemini"
            self.SYSTEM_ACCOUNT = "user"
            self.TEXT_KEY = "parts"


    def create_regex(self):
        single_float = "[0-9 .-]+,"
        reg = ""
        for _ in range(self.matrix_size[1]):
            reg += single_float

        self.reg = reg[:-1]


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
        while self.tokens > self.token_limit:
            # keeping the system prompt and never removing that.
            self.remove_llm_conversation(1)

        response = query_llm(self.client, self.llm_model_name, self.llm_conversation)
        return response


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

        updated_matrix = self.parse_parameters(matrix_response_with_reasoning)

        trial = 0
        while trial < 3 and len(updated_matrix) != self.matrix_size[0]:
            print("Could not parse matrix once, trying again", trial, updated_matrix)
            self.add_llm_conversation(
                self.llm_output_conversion_template.render(),
                "user"
            )
            response = self.query_llm()
            updated_matrix = self.parse_parameters(response)
            #self.add_llm_conversation(response, "assistant")
            # there is no need to keep the communication
            self.remove_llm_conversation(-1)
            trial += 1

        self.add_llm_conversation(matrix_response_with_reasoning, "assistant")
        return updated_matrix, matrix_response_with_reasoning


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


    def get_token_count(self, message):
        return len(self.encoder.encode(message))
