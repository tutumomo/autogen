# %% [markdown]
# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_planning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Auto Generated Agent Chat: Collaborative Task Solving with Coding and Planning Agent
# 
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use multiple agents to work together and accomplish a task that requires finding info from the web and coding. `AssistantAgent` is an LLM-based agent that can write and debug Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for a user to execute the code written by `AssistantAgent`. We further create a planning agent for the assistant agent to consult. The planning agent is a variation of the LLM-based `AssistantAgent` with a different system message.
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
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file. It first looks for an environment variable with a specified name. The value of the environment variable needs to be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by filter_dict.
# 
# It's OK to have only the OpenAI API key, or only the Azure OpenAI API key + base.
# 

# %%
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

# %% [markdown]
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-4',
#         'api_key': '<your OpenAI API key here>',
#     },  # OpenAI API endpoint for gpt-4
#     {
#         'model': 'gpt-4',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },  # Azure OpenAI API endpoint for gpt-4
#     {
#         'model': 'gpt-4-32k',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },  # Azure OpenAI API endpoint for gpt-4-32k
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods.
# 
# ## Construct Agents
# 
# We construct the planning agent named "planner" and a user proxy agent for the planner named "planner_user". We specify `human_input_mode` as "NEVER" in the user proxy agent, which will never ask for human feedback. We define `ask_planner` function to send a message to the planner and return the suggestion from the planner.

# %%
planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step that can be implemented by writing code. For example, browsing the web can be implemented by writing code that reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix.",
)
planner_user = autogen.UserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

# %% [markdown]
# We construct the assistant agent and the user proxy agent. We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for feedback when it receives a "TERMINATE" signal from the assistant agent. We set the `functions` in `AssistantAgent` and `function_map` in `UserProxyAgent` to use the created `ask_planner` function.

# %%
# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    # is_termination_msg=lambda x: "content" in x and x["content"] is not None and x["content"].rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "planning",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    function_map={"ask_planner": ask_planner},
)

# %% [markdown]
# ## Perform a task
# 
# We invoke the `initiate_chat()` method of the user proxy agent to start the conversation. When you run the cell below, you will be prompted to provide feedback after the assistant agent sends a "TERMINATE" signal at the end of the message. If you don't provide any feedback (by pressing Enter directly), the conversation will finish. Before the "TERMINATE" signal, the user proxy agent will try to execute the code suggested by the assistant agent on behalf of the user.

# %%
# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""Suggest a fix to an open good first issue of flaml""",
)

# %% [markdown]
# When the assistant needs to consult the planner, it suggests a function call to `ask_planner`. When this happens, a line like the following will be displayed:
# 
# ***** Suggested function Call: ask_planner *****
# 


