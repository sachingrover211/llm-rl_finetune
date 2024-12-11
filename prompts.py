SYSTEM_PROMPT = "You are guiding a robot across a grid. The size of the grid is {rows}by" + \
                "{cols} starting from indices 0 to {row_1} and 0 to {col_1}. Robot is " + \
                "supposed to navigate to the Goal " + \
                "location. Cell is defined using (x, y) location, where x specifies the row " + \
                "and y specifies column of the grid. Grid can also have multiple inaccessible " + \
                "cells and the robot should navigate around them to while moving around the grid."
SYSTEM_PROMPT_PACKAGE = "You are guiding a robot across a grid. The size of the grid is {rows}" + \
                "by{cols} starting from indices 0 to {row_1} and 0 to {col_1}. Robot is " + \
                "supposed to pickup packages from specific cells and then drop them to Goal " + \
                "location. Cell is defined using (x, y) location, where x specifies the row " + \
                "and y specifies column of the grid. Grid can also have multiple inaccessible " + \
                "cells and the robot should navigate around them to while moving around the grid."
ROBOT_CAPABILITIES = "Robot is capable of moving across grid, by walking to adjacent cells " + \
                "by moving up -- decreases x value to x-1, moving left -- decreases y value " + \
                "to y-1, moving right -- increase y value to y+1 and moving down -- increase " + \
                "x value to x+1."
ROBOT_CAPABILITIES_PACKAGE = "The robot is also capable of picking up packages when it " + \
                "reaches their location. It can also put them down at any open cell, " + \
                "i.e. the cells are not blocked off."
ROBOT_CAPABILITIES_TRACE = "Robot is capable of moving across grid, by walking to adjacent " + \
                "cells by performing action A -- moves the robot from (3, 1) to (2, 1), B " + \
                "moves from (3, 1) to (3, 0), C -- moves from (3, 1) to (3, 2) and D -- moves " + \
                "from (3, 1) to (4, 1)."
ROBOT_CAPABILITIES_TRACE_PACKAGE = "The robot is also capable of performing action E " + \
                "to pick packages when it reaches it's location, and perform action F to " + \
                "put them down at any open cell, i.e. the cells are not blocked off."
ROBOT_CAPABILITIES_LOG = "All the actions of the Robot are presented in the log of a trace " + \
                "in the table below. Please go through the log carefully to understand " + \
                "everything robot can do in the grid."
EXAMPLE_GRID = "*For example* there is a 5by5 grid, x can take values between 0 to 4, and " + \
                "similarly y can take values from 0 to 4. If the agent is (0, 0) " + \
                "and the goal location is (3, 4).\n" + \
                "Then the agent should perform Move Down to (1, 0), Move Down to (2, 0), " + \
                "and then continue to navigate to the Goal. " + \
                "Please ensure that the agent navigates around the blocked nodes."
EXAMPLE_GRID_PACKAGE = "*For example* there is a 5by5 grid, x can take values between 0 to 4, " + \
                "and similarly y can take values from 0 to 4. If the agent is (0, 0) and the " + \
                "package is at (2, 0), and the goal location is (3, 4).\n" + \
                "Then the agent should perform Move Down to (1, 0), Move Down to (2, 0), " + \
                "PickUp Package, and then continue to navigate to the Goal. " + \
                "Please ensure that the agent navigates around the blocked nodes."
EXAMPLE_GRID_TRACE = "*For example* there is a 5by5 grid, x can take values between " + \
                "0 to 4, and similarly y can take values from 0 to 4. If the agent is at " + \
                "(0, 0) and the package is at (2, 0), and the goal location is (3, 4).\n" + \
                "Then the agent should perform action D to goto (1, 0), followed by " + \
                "C to goto (1, 1), and then again D to goto (2, 1). Similarly it continue " + \
                "navigation to reach the goal while staying away from blocked cells."
EXAMPLE_GRID_TRACE_PACKAGE = "*For example* there is a 5by5 grid, x can take values between " + \
                "0 to 4, and similarly y can take values from 0 to 4. If the agent is at " + \
                "(0, 0) and the package is at (2, 0), and the goal location is (3, 4).\n" + \
                "Then the agent should perform action D to goto (1, 0), followed by action D " + \
                "to goto (2, 0), PickUp Package, and then continue to navigate to the Goal. " + \
                "Please ensure that the agent navigates around the blocked cells."
USER_PROMPT = "In the current Grid the Agent is at location {agent}. The agent has " + \
                "to navigate to {goal}. " + \
                "Currently {number_blocked} cells are blocked at locations {blocked_locs}."
USER_PROMPT_PACKAGE = "In the current Grid the Agent is at location {agent}. The package is " + \
                "available at {package}. The package has to be dropped off at {goal}. " + \
                "Currently {number_blocked} cells are blocked at locations {blocked_locs}."
PLAN_GENERATION = "Generate a plan for the robot to navigate the grid, by providing " + \
                "each action such as Move Up or Move Down. Please provide complete plan with " + \
                "one action in one line."
PLAN_GENERATION_TRACE = "Generate a plan for the robot to navigate the grid, by " + \
                "providing the action names to navigate and complete the task. Please " + \
                "provide one action in one line."
PLAN_VALIDATION_PROMPT = "Please evaluate the plan by performing each action in the " + \
                "environment, {plan}, to check whether it will work in the current scenario."
INTERNAL_MANIPULATION_SUGGESTION = "Internally you should track the agent location to " + \
                "evaluate correctness of the plan you are returning. For final response ONLY " + \
                "provide the plan with one action in one line. Do NOT add your analysis to it."
SI_PROMPT = "The plan failed. Please find the execution trace below in the table, showing " + \
                "the current location of the Agent execution action suggested by you. " + \
                "Can you suggest a new plan for the agent to reach the goal location of " + \
                "{goal}. Note the holes are at location {holes}."

COMPLETE_PROMPT = "{grid_description}\n\n{robot_actions}\n\n{example}"
COMPLETE_USER_PROMPT = "{current_grid}\n{final_question}\n{suggestion}"
