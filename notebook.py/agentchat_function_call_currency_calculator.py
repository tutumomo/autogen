# %% [markdown]
# # Currency Calculator: Task Solving with Provided Tools as Functions
# 

# %% [markdown]
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to make function calls with the new feature of OpenAI models (in model version 0613). A specified prompt and function configs must be passed to `AssistantAgent` to initialize the agent. The corresponding functions must be passed to `UserProxyAgent`, which will execute any function calls made by `AssistantAgent`. Besides this requirement of matching descriptions with functions, we recommend checking the system message in the `AssistantAgent` to ensure the instructions align with the function call descriptions.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install `pyautogen`:
# ```bash
# pip install pyautogen
# ```

# %%
# %pip install "pyautogen>=0.2.3"

# %% [markdown]
# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

# %%
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated

import autogen
from autogen.cache import Cache

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"tags": ["3.5-tool"]},  # comment out to get all
)

# %% [markdown]
# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by tags (you can filter by other keys as well). Only the configs with matching tags are kept in the list based on the filter condition.
# 
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-3.5-turbo',
#         'api_key': '<your OpenAI API key here>',
#         'tags': ['tool', '3.5-tool'],
#     },
#     {
#         'model': 'gpt-3.5-turbo',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#         'tags': ['tool', '3.5-tool'],
#     },
#     {
#         'model': 'gpt-3.5-turbo-16k',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#         'tags': ['tool', '3.5-tool'],
#     },
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

# %% [markdown]
# ## Making Function Calls
# 
# In this example, we demonstrate function call execution with `AssistantAgent` and `UserProxyAgent`. With the default system prompt of `AssistantAgent`, we allow the LLM assistant to perform tasks with code, and the `UserProxyAgent` would extract code blocks from the LLM response and execute them. With the new "function_call" feature, we define functions and specify the description of the function in the OpenAI config for the `AssistantAgent`. Then we register the functions in `UserProxyAgent`.
# 

# %%
llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)


CurrencySymbol = Literal["USD", "EUR"]


def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")


@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"

# %% [markdown]
# The decorator `@chatbot.register_for_llm()` reads the annotated signature of the function `currency_calculator` and generates the following JSON schema used by OpenAI API to suggest calling the function. We can check the JSON schema generated as follows:

# %%
chatbot.llm_config["tools"]

# %% [markdown]
# The decorator `@user_proxy.register_for_execution()` maps the name of the function to be proposed by OpenAI API to the actual implementation. The function mapped is wrapped since we also automatically handle serialization of the output of function as follows:
# 
# - string are untouched, and
# 
# - objects of the Pydantic BaseModel type are serialized to JSON.
# 
# We can check the correctness of of function map by using `._origin` property of the wrapped function as follows:

# %%
assert user_proxy.function_map["currency_calculator"]._origin == currency_calculator

# %% [markdown]
# Finally, we can use this function to accurately calculate exchange amounts:

# %%
with Cache.disk() as cache:
    # start the conversation
    res = user_proxy.initiate_chat(
        chatbot, message="How much is 123.45 USD in EUR?", summary_method="reflection_with_llm", cache=cache
    )

# %%
print("Chat summary:", res.summary)

# %% [markdown]
# ### Pydantic models

# %% [markdown]
# We can also use Pydantic Base models to rewrite the function as follows:

# %%
llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)


class Currency(BaseModel):
    currency: Annotated[CurrencySymbol, Field(..., description="Currency symbol")]
    amount: Annotated[float, Field(0, description="Amount of currency", ge=0)]


# another way to register a function is to use register_function instead of register_for_execution and register_for_llm decorators
def currency_calculator(
    base: Annotated[Currency, "Base currency: amount and currency symbol"],
    quote_currency: Annotated[CurrencySymbol, "Quote currency symbol"] = "USD",
) -> Currency:
    quote_amount = exchange_rate(base.currency, quote_currency) * base.amount
    return Currency(amount=quote_amount, currency=quote_currency)


autogen.agentchat.register_function(
    currency_calculator,
    caller=chatbot,
    executor=user_proxy,
    description="Currency exchange calculator.",
)

# %%
chatbot.llm_config["tools"]

# %%
with Cache.disk() as cache:
    # start the conversation
    res = user_proxy.initiate_chat(
        chatbot, message="How much is 112.23 Euros in US Dollars?", summary_method="reflection_with_llm", cache=cache
    )

# %%
print("Chat summary:", res.summary)

# %%
with Cache.disk() as cache:
    # start the conversation
    res = user_proxy.initiate_chat(
        chatbot,
        message="How much is 123.45 US Dollars in Euros?",
        cache=cache,
    )

# %%
print("Chat history:", res.chat_history)


