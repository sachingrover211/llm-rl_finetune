import gymnasium as gym
from openai import OpenAI
import random
import numpy as np
import os
import time
from jinja2 import Template


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.client = OpenAI()
        self.llm_conversation = []
        assert llm_model_name in ["o1-preview", "gpt-4o"]
        self.llm_model_name = llm_model_name

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

    def llm_update_q_table(self, q_table, replay_buffer, parse_q_table):
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

        new_q_table_list = parse_q_table(new_q_table)

        return new_q_table_list, [new_q_table_with_reasoning, new_q_table]
