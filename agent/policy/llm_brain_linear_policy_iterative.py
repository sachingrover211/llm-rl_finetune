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
        llm_1st_round_template: Template,
        llm_round_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_1st_round_template = llm_1st_round_template
        self.llm_round_template = llm_round_template
        self.client = OpenAI()
        self.llm_conversation = []
        assert llm_model_name in ["gpt-4o", "o1-preview", "o1"]
        self.llm_model_name = llm_model_name

    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})

    def reset_llm_conversation(self):
        self.llm_conversation = []
        system_prompt = self.llm_si_template.render()
        self.add_llm_conversation(system_prompt, "system")

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
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(
        self,
        last_performance="",
        reasoning="",
        replay_buffer=None,
        parse_parameters=None,
        first_round=False,
        iter_idx=0,
    ):

        if first_round:

            query = self.llm_1st_round_template.render(
                replay_buffer=replay_buffer,
            )
        else:

            query = self.llm_round_template.render(
                last_performance=last_performance,
                reasoning=reasoning,
                replay_buffer=replay_buffer,
                iter=iter_idx,
            )

        self.add_llm_conversation(query, "user")
        print("Querying the LLM...\n")
        print(query)
        response = self.query_llm()
        print(response)
        print("Parsing the response...")

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(response)
        else:
            new_parameters_list = parse_parameters(response)

        return new_parameters_list, query, response
