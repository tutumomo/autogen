# %% [markdown]
# # StateFlow: Build Workflows through State-Oriented Actions
# 
# AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. In this notebook, we introduce how to use groupchat to build workflows with AutoGen agents from a state-oriented perspective.
# 
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

# %% [markdown]
# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

# %%
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "tags": ["gpt-4", "gpt-4-32k"],
    },
)

# %% [markdown]
# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
# :::
# ````
# 
# ## A workflow for research
# 
# <figure>
#     <img src="../website/blog/2024-02-29-StateFlow/img/sf_example_1.png"  width="700"
#          alt="SF_Example_1">
#     </img>
# </figure>
# 
# We define the following agents:
# - Initializer: Start the workflow by sending a task.
# - Coder: Retrieve papers from the internet by writing code.
# - Executor: Execute the code.
# - Scientist: Read the papers and write a summary.
# 
# 
# In the Figure, we define a simple workflow for research with 4 states: Init, Retrieve, Reserach and End. Within each state, we will call different agents to perform the tasks.
# - Init: We use the initializer to start the workflow.
# - Retrieve: We will first call the coder to write code and then call the executor to execute the code.
# - Research: We will call the scientist to read the papers and write a summary.
# - End: We will end the workflow.
# 
# Through customizing the speaker selection method, we can easily realize the state-oriented workflow by defining the transitions between different agents.

# %%
import tempfile

from autogen.coding import LocalCommandLineCodeExecutor

temp_dir = tempfile.TemporaryDirectory()
executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
)

gpt4_config = {
    "cache_seed": False,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False,
)


coder = autogen.AssistantAgent(
    name="Retrieve_Action_1",
    llm_config=gpt4_config,
    system_message="""You are the Coder. Given a topic, write code to retrieve related papers from the arXiv API, print their title, authors, abstract, and link.
You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
executor = autogen.UserProxyAgent(
    name="Retrieve_Action_2",
    system_message="Executor. Execute the code written by the Coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"executor": executor},
)
scientist = autogen.AssistantAgent(
    name="Research_Action_1",
    llm_config=gpt4_config,
    system_message="""You are the Scientist. Please categorize papers after seeing their abstracts printed and create a markdown table with Domain, Title, Authors, Summary and Link""",
)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        # init -> retrieve
        return coder
    elif last_speaker is coder:
        # retrieve: action 1 -> action 2
        return executor
    elif last_speaker is executor:
        if messages[-1]["content"] == "exitcode: 1":
            # retrieve --(execution failed)--> retrieve
            return coder
        else:
            # retrieve --(execution sucess)--> research
            return scientist
    elif last_speaker == "Scientist":
        # research -> end
        return None


groupchat = autogen.GroupChat(
    agents=[initializer, coder, executor, scientist],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# %%
chat_result = initializer.initiate_chat(
    manager, message="Topic: LLM applications papers from last week. Requirement: 5 - 10 papers from different domains."
)


