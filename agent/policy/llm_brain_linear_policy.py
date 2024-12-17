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

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [float(x.strip().strip(',')) for x in row.split(",")]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"replay_buffer_string": str(replay_buffer), "parameters_string": str(parameters)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [new_parameters_with_reasoning, new_parameters]
