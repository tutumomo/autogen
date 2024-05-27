# %% [markdown]
# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/oai_chatgpt_gpt4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved. 
# 
# Licensed under the MIT License.
# 
# # Use AutoGen to Tune ChatGPT
# 
# AutoGen offers a cost-effective hyperparameter optimization technique [EcoOptiGen](https://arxiv.org/abs/2303.04673) for tuning Large Language Models. The study finds that tuning hyperparameters can significantly improve the utility of LLMs.
# Please find documentation about this feature [here](/docs/Use-Cases/AutoGen#enhanced-inference).
# 
# In this notebook, we tune OpenAI ChatGPT (both GPT-3.5 and GPT-4) models for math problem solving. We use [the MATH benchmark](https://crfm.stanford.edu/helm/latest/?group=math_chain_of_thought) for measuring mathematical problem solving on competition math problems with chain-of-thoughts style reasoning.
# 
# Related link: [Blogpost](https://microsoft.github.io/autogen/blog/2023/04/21/LLM-tuning-math) based on this experiment.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install with the [blendsearch] option:
# ```bash
# pip install "pyautogen[blendsearch]"
# ```

# %%
# %pip install "pyautogen[blendsearch]<0.2" datasets

# %% [markdown]
# AutoGen has provided an API for hyperparameter optimization of OpenAI ChatGPT models: `autogen.ChatCompletion.tune` and to make a request with the tuned config: `autogen.ChatCompletion.create`. First, we import autogen:

# %%
import logging

import datasets

import autogen
from autogen.math_utils import eval_math_responses

# %% [markdown]
# ### Set your API Endpoint
# 
# The [`config_list_openai_aoai`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_openai_aoai) function tries to create a list of  Azure OpenAI endpoints and OpenAI endpoints. It assumes the api keys and api bases are stored in the corresponding environment variables or local txt files:
# 
# - OpenAI API key: os.environ["OPENAI_API_KEY"] or `openai_api_key_file="key_openai.txt"`.
# - Azure OpenAI API key: os.environ["AZURE_OPENAI_API_KEY"] or `aoai_api_key_file="key_aoai.txt"`. Multiple keys can be stored, one per line.
# - Azure OpenAI API base: os.environ["AZURE_OPENAI_API_BASE"] or `aoai_api_base_file="base_aoai.txt"`. Multiple bases can be stored, one per line.
# 
# It's OK to have only the OpenAI API key, or only the Azure OpenAI API key + base.
# 

# %%
config_list = autogen.config_list_openai_aoai()

# %% [markdown]
# The config list looks like the following:
# ```python
# config_list = [
#     {'api_key': '<your OpenAI API key here>'},  # only if OpenAI API key is found
#     {
#         'api_key': '<your first Azure OpenAI API key here>',
#         'base_url': '<your first Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },  # only if at least one Azure OpenAI API key is found
#     {
#         'api_key': '<your second Azure OpenAI API key here>',
#         'base_url': '<your second Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },  # only if the second Azure OpenAI API key is found
# ]
# ```
# 
# You can directly override it if the above function returns an empty list, i.e., it doesn't find the keys in the specified locations.

# %% [markdown]
# ## Load dataset
# 
# We load the competition_math dataset. The dataset contains 201 "Level 2" Algebra examples. We use a random sample of 20 examples for tuning the generation hyperparameters and the remaining for evaluation.

# %%
seed = 41
data = datasets.load_dataset("competition_math")
train_data = data["train"].shuffle(seed=seed)
test_data = data["test"].shuffle(seed=seed)
n_tune_data = 20
tune_data = [
    {
        "problem": train_data[x]["problem"],
        "solution": train_data[x]["solution"],
    }
    for x in range(len(train_data))
    if train_data[x]["level"] == "Level 2" and train_data[x]["type"] == "Algebra"
][:n_tune_data]
test_data = [
    {
        "problem": test_data[x]["problem"],
        "solution": test_data[x]["solution"],
    }
    for x in range(len(test_data))
    if test_data[x]["level"] == "Level 2" and test_data[x]["type"] == "Algebra"
]
print(len(tune_data), len(test_data))

# %% [markdown]
# Check a tuning example:

# %%
print(tune_data[1]["problem"])

# %% [markdown]
# Here is one example of the canonical solution:

# %%
print(tune_data[1]["solution"])

# %% [markdown]
# ## Define Success Metric
# 
# Before we start tuning, we must define the success metric we want to optimize. For each math task, we use voting to select a response with the most common answers out of all the generated responses. We consider the task successfully solved if it has an equivalent answer to the canonical solution. Then we can optimize the mean success rate of a collection of tasks.

# %% [markdown]
# ## Use the tuning data to find a good configuration
# 

# %% [markdown]
# For (local) reproducibility and cost efficiency, we cache responses from OpenAI with a controllable seed.

# %%
autogen.ChatCompletion.set_cache(seed)

# %% [markdown]
# This will create a disk cache in ".cache/{seed}". You can change `cache_path_root` from ".cache" to a different path in `set_cache()`. The cache for different seeds are stored separately.
# 
# ### Perform tuning
# 
# The tuning will take a while to finish, depending on the optimization budget. The tuning will be performed under the specified optimization budgets.
# 
# * `inference_budget` is the benchmark's target average inference budget per instance. For example, 0.004 means the target inference budget is 0.004 dollars, which translates to 2000 tokens (input + output combined) if the gpt-3.5-turbo model is used.
# * `optimization_budget` is the total budget allowed for tuning. For example, 1 means 1 dollar is allowed in total, which translates to 500K tokens for the gpt-3.5-turbo model.
# * `num_sumples` is the number of different hyperparameter configurations allowed to be tried. The tuning will stop after either num_samples trials are completed or optimization_budget dollars are spent, whichever happens first. -1 means no hard restriction in the number of trials and the actual number is decided by `optimization_budget`.
# 
# Users can specify tuning data, optimization metric, optimization mode, evaluation function, search spaces etc.. The default search space is:
# 
# ```python
# default_search_space = {
#     "model": tune.choice([
#         "gpt-3.5-turbo",
#         "gpt-4",
#     ]),
#     "temperature_or_top_p": tune.choice(
#         [
#             {"temperature": tune.uniform(0, 2)},
#             {"top_p": tune.uniform(0, 1)},
#         ]
#     ),
#     "max_tokens": tune.lograndint(50, 1000),
#     "n": tune.randint(1, 100),
#     "prompt": "{prompt}",
# }
# ```
# 
# The default search space can be overridden by users' input.
# For example, the following code specifies a fixed prompt template. The default search space will be used for hyperparameters that don't appear in users' input.

# %%
prompts = [
    "{problem} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}."
]
config, analysis = autogen.ChatCompletion.tune(
    data=tune_data,  # the data for tuning
    metric="success_vote",  # the metric to optimize
    mode="max",  # the optimization mode
    eval_func=eval_math_responses,  # the evaluation function to return the success metrics
    # log_file_name="logs/math.log",  # the log file name
    inference_budget=0.02,  # the inference budget (dollar per instance)
    optimization_budget=1,  # the optimization budget (dollar in total)
    # num_samples can further limit the number of trials for different hyperparameter configurations;
    # -1 means decided by the optimization budget only
    num_samples=20,
    model="gpt-3.5-turbo",  # comment to tune both gpt-3.5-turbo and gpt-4
    prompt=prompts,  # the prompt templates to choose from
    # stop="###",  # the stop sequence
    config_list=config_list,  # the endpoint list
    allow_format_str_template=True,  # whether to allow format string template
    # logging_level=logging.INFO,  # the logging level
)

# %% [markdown]
# ### Output tuning results
# 
# After the tuning, we can print out the config and the result found by AutoGen, which uses flaml for tuning.

# %%
print("optimized config", config)
print("best result on tuning data", analysis.best_result)

# %% [markdown]
# ### Make a request with the tuned config
# 
# We can apply the tuned config on the request for an example task:

# %%
response = autogen.ChatCompletion.create(context=tune_data[1], config_list=config_list, **config)
metric_results = eval_math_responses(autogen.ChatCompletion.extract_text(response), **tune_data[1])
print("response on an example data instance:", response)
print("metric_results on the example data instance:", metric_results)

# %% [markdown]
# ### Evaluate the success rate on the test data
# 
# You can use `autogen.ChatCompletion.test` to evaluate the performance of an entire dataset with the tuned config. The following code will take a while (30 mins to 1 hour) to evaluate all the test data instances if uncommented and run. It will cost roughly $3. 

# %%
# result = autogen.ChatCompletion.test(test_data, logging_level=logging.INFO, config_list=config_list, **config)
# print("performance on test data with the tuned config:", result)

# %% [markdown]
# What about the default, untuned gpt-4 config (with the same prompt as the tuned config)? We can evaluate it and compare:

# %%
# the following code will cost roughly $2 if uncommented and run.

# default_config = {"model": 'gpt-4', "prompt": prompts[0], "allow_format_str_template": True}
# default_result = autogen.ChatCompletion.test(test_data, config_list=config_list, **default_config)
# print("performance on test data from gpt-4 with a default config:", default_result)

# %%
# print("tuned config succeeds in {:.1f}% test cases".format(result["success_vote"] * 100))
# print("untuned config succeeds in {:.1f}% test cases".format(default_result["success_vote"] * 100))

# %% [markdown]
# The default use of GPT-4 has a much lower accuracy. Note that the default config has a lower inference cost. What if we heuristically increase the number of responses n?

# %%
# The following evaluation costs $3 and longer than one hour if you uncomment it and run it.

# config_n2 = {"model": 'gpt-4', "prompt": prompts[0], "n": 2, "allow_format_str_template": True}
# result_n2 = autogen.ChatCompletion.test(test_data, config_list=config_list, **config_n2)
# print("performance on test data from gpt-4 with a default config and n=2:", result_n2)

# %% [markdown]
# The inference cost is doubled and matches the tuned config. But the success rate doesn't improve much. What if we further increase the number of responses n to 5?

# %%
# The following evaluation costs $8 and longer than one hour if you uncomment it and run it.

# config_n5 = {"model": 'gpt-4', "prompt": prompts[0], "n": 5, "allow_format_str_template": True}
# result_n5 = autogen.ChatCompletion.test(test_data, config_list=config_list, **config_n5)
# print("performance on test data from gpt-4 with a default config and n=5:", result_n5)

# %% [markdown]
# We find that the 'success_vote' metric is increased at the cost of exceeding the inference budget. But the tuned configuration has both higher 'success_vote' (91% vs. 87%) and lower average inference cost ($0.015 vs. $0.037 per instance).
# 
# A developer could use AutoGen to tune the configuration to satisfy the target inference budget while maximizing the value out of it.


