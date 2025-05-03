import re
import numpy as np
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.linear_policy import LinearPolicy


class BaseFinetuneAgent:
    def __init__(
        self,
        actions,
        states,
        num_evaluation_episodes,
    ):
        self.num_evaluation_episodes = num_evaluation_episodes
        self.policy = LinearPolicy(states, actions)
        self.matrix_size = (states + 1, actions)
        self.rank = (states+1)*actions


    def initialize_policy(self, weight):
        self.policy.initialize_policy_with_weights(weight)


    def rollout_episode(self, world):
        state = world.reset()

        done = False
        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done = world.step(action)
            state = next_state

        return world.get_accu_reward()


    def evaluate_policy(self, world):
        results = []

        for idx in range(self.num_evaluation_episodes):
            result = self.rollout_episode(world)
            results.append(result)
        return results


    def parse_response(self, response):
        new_parameters_list = list()
        new_rs = response.lower()

        if "<policy>" in new_rs:
            new_rs = [self.get_sub_string(new_rs, "<policy>", "</policy>")]
        else:
            new_rs = response.split("\n")

        result = list()
        for r in new_rs:
            matches = self.pattern.findall(r)
            if len(matches) == self.rank:
                for match in matches:
                    result.append(float(match[1]))

        return np.array(result).reshape(-1)


    def get_sub_string(self, response, start_tag, end_tag):
        start_index = response.find(start_tag)
        if start_index == -1:
            return ""

        end_index = response.find(end_tag)
        if end_index == -1:
            return ""

        start_index += len(start_tag)
        return response[start_index: end_index]
