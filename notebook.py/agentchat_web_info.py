# %% [markdown]
# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_web_info.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Auto Generated Agent Chat: Solving Tasks Requiring Web Info
# 
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to perform tasks which require acquiring info from the web:
# * discuss a paper based on its URL.
# * discuss about the stock market.
# 
# Here `AssistantAgent` is an LLM-based agent that can write Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for a user to execute the code written by `AssistantAgent`. By setting `human_input_mode` properly, the `UserProxyAgent` can also prompt the user for feedback to `AssistantAgent`. For example, when `human_input_mode` is set to "TERMINATE", the `UserProxyAgent` will execute the code written by `AssistantAgent` directly and return the execution results (success or failure and corresponding outputs) to `AssistantAgent`, and prompt the user for feedback when the task is finished. When user feedback is provided, the `UserProxyAgent` will directly pass the feedback to `AssistantAgent`.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install pyautogen and docker:
# ```bash
# pip install pyautogen docker
# ```

# %%
# %pip install "pyautogen>=0.2.3" docker

# %% [markdown]
# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# %%
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

# %% [markdown]
# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). Only the models with matching names are kept in the list based on the filter condition.
# 
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-4-32k',
#         'api_key': '<your OpenAI API key here>',
#     },
#     {
#         'model': 'gpt4',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
#     {
#         'model': 'gpt-4-32k-0314',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods.

# %% [markdown]
# ## Construct Agents
# 
# We construct the assistant agent and the user proxy agent. We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for human feedback when it receives a "TERMINATE" signal from the assistant agent.

# %%
# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

# %% [markdown]
# ## Example Task: Paper Talk from URL
# 
# We invoke the `initiate_chat()` method of the user proxy agent to start the conversation. When you run the cell below, you will be prompted to provide feedback after the assistant agent sends a "TERMINATE" signal at the end of the message. If you don't provide any feedback (by pressing Enter directly), the conversation will finish. Before the "TERMINATE" signal, the user proxy agent will try to execute the code suggested by the assistant agent on behalf of the user.

# %%
# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
Who should read this paper: https://arxiv.org/abs/2308.08155
""",
)

# %% [markdown]
# ## Example Task: Chat about Stock Market

# %%
user_proxy.initiate_chat(
    assistant,
    message="""Show me the YTD gain of 10 largest technology companies as of today.""",
)


