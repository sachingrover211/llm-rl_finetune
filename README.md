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
* llm\_model\_name: model used for getting the response. Linear policy can work with `gpt-4o`. Q-table could not be created properly requires `o1`. OFFLINE models should provide path. Other options possible are HF models which should provide the HF name for downloading the model.  
* num\_evaluation\_episodes: number of episodes per evaluation.
* record\_video: (true/false) record videos for the evaluation episode. Currently not working correctly.
* use\_replay\_buffer: (true/false) should prompt use replay buffer or not.
* reset\_llm\_conversation: (true/false) after every episode should we reset the replay buffer.

Then run the command

```python main.py --config <path to config>```

Currently it supports finetuning the model using `finetune.py`. There are two slurm files that were used for finetuning -- `finetune.sh` and inference -- `inference.sh`. The specific models to finetune can be set as a property at the finetuning file, as well as the logging directory. Inference is run through config files. It is similar to the previous command however, the model is defined as `OFFLINE`, with the location provided in the llm_model_name.
