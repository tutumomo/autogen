# %% [markdown]
# # Task Solving with Provided Tools as Functions (Asynchronous Function Calls)
# 

# %% [markdown]
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to make function calls with the new feature of OpenAI models (in model version 0613). A specified prompt and function configs must be passed to `AssistantAgent` to initialize the agent. The corresponding functions must be passed to `UserProxyAgent`, which will execute any function calls made by `AssistantAgent`. Besides this requirement of matching descriptions with functions, we recommend checking the system message in the `AssistantAgent` to ensure the instructions align with the function call descriptions.
# 
# ````{=mdx}
# :::info Requirements
# Install `pyautogen`:
# ```bash
# pip install pyautogen
# ```
# 
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````
# 

# %%
import time

from typing_extensions import Annotated

import autogen
from autogen.cache import Cache

config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={"tags": ["tool"]})

# %% [markdown]
# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````
# 
# ## Making Async and Sync Function Calls
# 
# In this example, we demonstrate function call execution with `AssistantAgent` and `UserProxyAgent`. With the default system prompt of `AssistantAgent`, we allow the LLM assistant to perform tasks with code, and the `UserProxyAgent` would extract code blocks from the LLM response and execute them. With the new "function_call" feature, we define functions and specify the description of the function in the OpenAI config for the `AssistantAgent`. Then we register the functions in `UserProxyAgent`.

# %%
llm_config = {
    "config_list": config_list,
}

coder = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. You have a stopwatch and a timer, these tools can and should be used in parallel. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A proxy for the user for executing code.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"},
)

# define functions according to the function description

# An example async function registered using register_for_llm and register_for_execution decorators


@user_proxy.register_for_execution()
@coder.register_for_llm(description="create a timer for N seconds")
async def timer(num_seconds: Annotated[str, "Number of seconds in the timer."]) -> str:
    for i in range(int(num_seconds)):
        time.sleep(1)
        # should print to stdout
    return "Timer is done!"


# An example sync function registered using register_function
def stopwatch(num_seconds: Annotated[str, "Number of seconds in the stopwatch."]) -> str:
    for i in range(int(num_seconds)):
        time.sleep(1)
    return "Stopwatch is done!"


autogen.agentchat.register_function(
    stopwatch,
    caller=coder,
    executor=user_proxy,
    description="create a stopwatch for N seconds",
)

# %% [markdown]
# Start the conversation. `await` is used to pause and resume code execution for async IO operations. Without `await`, an async function returns a coroutine object but doesn't execute the function. With `await`, the async function is executed and the current function is paused until the awaited function returns a result.

# %%
with Cache.disk() as cache:
    await user_proxy.a_initiate_chat(  # noqa: F704
        coder,
        message="Create a timer for 5 seconds and then a stopwatch for 5 seconds.",
        cache=cache,
    )

# %% [markdown]
# # Async Function Call with Group Chat
# Sync and async can be used in topologies beyond two agents. Below, we show this feature for a group chat.

# %%
markdownagent = autogen.AssistantAgent(
    name="Markdown_agent",
    system_message="Respond in markdown only",
    llm_config=llm_config,
)

# Add a function for robust group chat termination


@user_proxy.register_for_execution()
@markdownagent.register_for_llm()
@coder.register_for_llm(description="terminate the group chat")
def terminate_group_chat(message: Annotated[str, "Message to be sent to the group chat."]) -> str:
    return f"[GROUPCHAT_TERMINATE] {message}"


groupchat = autogen.GroupChat(agents=[user_proxy, coder, markdownagent], messages=[], max_round=12)

llm_config_manager = llm_config.copy()
llm_config_manager.pop("functions", None)
llm_config_manager.pop("tools", None)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_manager,
    is_termination_msg=lambda x: "GROUPCHAT_TERMINATE" in x.get("content", ""),
)

# %% [markdown]
# Finally, we initialize the chat that would use the functions defined above:

# %%
message = """
1) Create a timer and a stopwatch for 5 seconds each in parallel.
2) Pretty print the result as md.
3) when 1 and 2 are done, terminate the group chat
"""

with Cache.disk() as cache:
    await user_proxy.a_initiate_chat(  # noqa: F704
        manager,
        message=message,
        cache=cache,
    )

# %%



