import gymnasium as gym
from openai import OpenAI
import random
import numpy as np
import os
import re
import time
from jinja2 import Template


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
        assert llm_model_name in ["o1-preview", "gpt-4o"]
        self.llm_model_name = llm_model_name
        self.is_first_query = True
        self.episode = 0

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})

    def query_llm(self):
        for attempt in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=self.llm_conversation,
                )
                # add the response to self.llm_conversation
                self.add_llm_conversation(
                    completion.choices[0].message.content, "assistant"
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    print("Waiting for 120 seconds before retrying...")
                    time.sleep(120)

    def parse_q_table(self, q_table_string):
        new_q_values_list = []

        # Update the Q-table based on the new Q-table
        for row in q_table_string.split("\n"):
            if row.strip():
                row = row.split("|")
                if len(row) == 4:
                    try:
                        position, velocity, action, q_value = row
                        position = int(position.strip())
                        velocity = int(velocity.strip())
                        action = int(action.strip())
                        q_value = float(q_value.strip())
                        new_q_values_list.append(
                            ((position, velocity), (action,), q_value)
                        )
                    except:
                        pass

        return new_q_values_list

    def llm_update_q_table(self, q_table, replay_buffer):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"replay_buffer_string": str(replay_buffer), "q_table_string": str(q_table)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_q_table_with_reasoning = self.query_llm()

        self.add_llm_conversation(new_q_table_with_reasoning, "assistant")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_q_table = self.query_llm()

        # print(f"New Q-table: {new_q_table}")

        new_q_table_list = self.parse_q_table(new_q_table)

        return new_q_table_list, new_q_table_with_reasoning


    def llm_update_linear_policy(self, linear_policy, average_reward, replay_buffer = None):
        #self.reset_llm_conversation()

        repeated_template = self.llm_ui_template if self.llm_ui_template else self.llm_si_template

        if repeated_template == self.llm_ui_template:
            system_prompt = self.llm_si_template.render()

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
        if self.is_first_query and repeated_template == self.llm_ui_template:
            self.add_llm_conversation(system_prompt, "system")
            self.is_first_query = False

        self.add_llm_conversation(repeated_prompt, "user")
        matrix_response_with_reasoning = self.query_llm()

        #print(matrix_response_with_reasoning)
        if replay_buffer:
            # in this case we do not want to track all replay buffers
            # thus we will not keep in llm_conversations
            self.llm_conversation.pop(-1)
            temp = repeated_template.render({
                "replay_buffer_string": "",
                "matrix_string": str(linear_policy),
                "reward": average_reward,
                "episode_number": self.episode
            })
            print(temp)
            self.add_llm_conversation(temp, "user")

        self.add_llm_conversation(matrix_response_with_reasoning, "assistant")
        updated_matrix = self.parse_parameters(matrix_response_with_reasoning)

        if len(updated_matrix) == 0:
            self.add_llm_conversation(
                self.llm_output_conversion_template.render(),
                "user",
            )
            response = self.query_llm()
            updated_matrix = self.parse_parameters(response)

            self.add_llm_conversation(response, "assistant")

        print(updated_matrix)
        return updated_matrix, matrix_response_with_reasoning


    def parse_parameters(self, parameters_string):
        new_parameters_list = []
        # this is for two dimensions, basically two actions
        # for greater than two actions we will have to re-write this
        # we can't do this individually as some numbers in the conversation
        # will also get parsed.
        reg = "[0-9 .-]+,[0-9 .-]+"

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    temp = re.findall(reg, row)
                    if len(temp) == 0:
                        continue
                    temp = temp[0]
                    parameters_row = [float(x.strip().strip(',')) for x in temp.strip().split(",")]
                    new_parameters_list.append(parameters_row)
                    if len(new_parameters_list) == 5:
                        break
                except Exception as e:
                    print(e)

        return new_parameters_list
