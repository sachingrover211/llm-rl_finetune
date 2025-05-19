# LLM RL Finetune
Policy improvement for RL using LLMs

### Installation
Make a virtual environment using conda or venv for Python 3.10. We have not tried with other variants of Python for now.

After activating the virtual environment run `pip install -r requirements.txt`

### Running the code

For running first define a config file, and their property description is defined below -

* task: name of the task which is asserted before running.
* num\_episodes: number of episodes in the experiment.
* gym\_env\_name: name used directly to instantiate the experiment.
* render\_mode: can be used to define the render mode for the environment.
* continuous: defines policy to be used to solve the environment. Is it a Q-table or Linear Policy.
* logdir: logging directory where logs will be stored
* actions: number of actions that will be used. For Q-table it is a list of actions.
* states: number of states. For Q-table this would be the list of numbers.
* max\_traj\_count: maximum trajectory for which replay buffer will be created.
* max\_traj\_length: maximum length of episode. For example in cartpole it is 500.
* template\_dir: templates for prompting. Most of them can be found in `agent/policy/templates`.
* llm\_si\_template\_name: system prompt template. Run once at the beginning of the experiment.
* llm\_ui\_template\_name: user prompt template, sent for optimizing parameters after every episode.
* llm\_output\_conversion\_template\_name: in case the output format does not match, then this prompt is used to correct the formatting.
* llm\_model\_name: model used for getting the response. Linear policy can work with `gpt-4o`. Q-table could not be created properly requires `o1`.  
* num\_evaluation\_episodes: number of episodes per evaluation.
* record\_video: (true/false) record videos for the evaluation episode. Currently not working correctly.
* use\_replay\_buffer: (true/false) should prompt use replay buffer or not.
* reset\_llm\_conversation: (true/false) after every episode should we reset the replay buffer.

Then run the command

```python main.py --config <path to config>```


Currently the code supports --

* Mountaincar -- discrete and continuous
* Cartpole
* Inverted pendulum (continuous single action version of cartpole)
* Hopper -- working on it
* Swimmer -- The results were not that great for this.

Frozen lake is forthcoming (it hasnt been tested for a little while the agent is not working right now).

For instructions to add a new domain, please reach out to the repo owner.
