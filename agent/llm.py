from agent.base_agent import Agent
from utils import llm_request
from properties import SYSTEM_PROMPT


class LLMAgent(Agent):
    def __init__(self):
        actions = [
            ""
        ]
        super().__init__(actions)
        self.conversation = list()
        self.plan = list()
        self.is_trace = True
        self.is_plan_generated = False
        self.is_navigation_only = True
        self.is_log = True
        self.init_trace = list()


    def next(self, grid):
        if len(self.plan) == 0 and not self.is_plan_generated:
            _plan = self.generate_plan(grid)
            self.plan = self.parse_plan(_plan)
            self.is_plan_generated = True

        if len(self.plan) > 0:
            return self.plan.pop(0)
        else:
            return None

    def generate_plan(self, grid):
        self.agent_loc = ", ".join([grid.get_location(loc) for loc in grid.state.get("agent")])
        self.goal_loc = ", ".join([grid.get_location(loc) for loc in grid.state.get("goal")])
        self.blocked_loc = ", ".join([grid.get_location(loc) for loc in grid.state.get("holes")])

        capabilities = ROBOT_CAPABILITIES
        grid_example = EXAMPLE_GRID
        plan_generation = PLAN_GENERATION
        if not self.is_navigation_only:
            capabilities += " " + ROBOT_CAPABILITIES_PACKAGE
            grid_example = EXAMPLE_GRID_PACKAGE

        if self.is_trace:
            capabilities = ROBOT_CAPABILITIES_TRACE
            grid_example = EXAMPLE_GRID_TRACE
            plan_generation = PLAN_GENERATION_TRACE
            if not self.is_navigation_only:
                capabilities += " " + ROBOT_CAPABILITIES_TRACE_PACKAGE
                grid_example = EXAMPLE_GRID_TRACE_PACKAGE

        if self.is_log:
            capabilities = ROBOT_CAPABILITIES_LOG

        sp = SYSTEM_PROMPT.format(
            rows = grid.rows,
            cols = grid.cols, 
            row_1 = grid.rows - 1,
            col_1 = grid.cols - 1
        )
        _prompt1 = COMPLETE_PROMPT.format(
            grid_description = sp,
            robot_actions = capabilities,
            example = grid_example
        )
        if self.is_log:
            _prompt1 +="\n\n" + "\n".join(self.init_trace)
 
        self.conversation.append({"role": "system", "content": _prompt1})

        up = USER_PROMPT.format(
            agent = self.agent_loc,
            #package = ", ".join([grid.get_location(loc) for loc in grid.state.get("packages")]),
            goal = self.goal_loc,
            number_blocked = len(grid.state.get("holes")),
            blocked_locs = self.blocked_loc
        )

        _prompt2 = COMPLETE_USER_PROMPT.format(
            current_grid = up,
            final_question = plan_generation,
            suggestion = INTERNAL_MANIPULATION_SUGGESTION
        )
        self.conversation.append({"role": "user", "content": _prompt2})
        print(_prompt1)
        print(_prompt2)

        message = get_response(self.conversation)
        response = message.content
        self.conversation.append({"role": "assistant", "content": response})
        print("######### LLM Response below")
        print(response.strip())

        return response.strip()


    def self_improve(self, trace):
        prompt = SI_PROMPT.format(
            goal = self.goal_loc,
            holes = self.blocked_loc
        )

        prompt += "\n\n" + "\n".join(trace)
        prompt += "\n" + INTERNAL_MANIPULATION_SUGGESTION
        print(prompt)
        self.conversation.append({"role": "user", "content": prompt})

        message = get_response(self.conversation)
        response = message.content
        self.conversation.append({"role": "assistant", "content": response})
        print("######### LLM Response below for self improvement")
        print(response.strip())

        self.plan = self.parse_plan(response.strip())


    def parse_plan(self, plan):
        plan = plan.split("\n")
        _actions = {
            "up": UpAction(),
            "left": LeftAction(),
            "right": RightAction(),
            "down": DownAction()
        }
        if self.is_trace:
            _actions = {
                "A": UpAction(),
                "B": LeftAction(),
                "C": RightAction(),
                "D": DownAction()
            }

        return_plan = list()
        for p in plan:
            p = p.strip()
            if not self.is_trace:
                p = p.split(" ")[1].lower()

            if p not in _actions:
                continue
            return_plan.append(_actions[p])

        return return_plan

