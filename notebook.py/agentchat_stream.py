# %% [markdown]
# <a href="https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_stream.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Interactive LLM Agent Dealing with Data Stream
# 
# AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
# 
# In this notebook, we demonstrate how to use customized agents to continuously acquire news from the web and ask for investment suggestions.
# 
# ## Requirements
# 
# AutoGen requires `Python>=3.8`. To run this notebook example, please install:
# ```bash
# pip install pyautogen
# ```

# %%
# %pip install "pyautogen>=0.2.3"

# %% [markdown]
# ## Set your API Endpoint
# 
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
# 

# %%
import asyncio

import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

# %% [markdown]
# It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well). Only the models with matching names are kept in the list based on the filter condition.
# 
# The config list looks like the following:
# ```python
# config_list = [
#     {
#         'model': 'gpt-4',
#         'api_key': '<your OpenAI API key here>',
#     },
#     {
#         'model': 'gpt-3.5-turbo',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
#     {
#         'model': 'gpt-3.5-turbo-16k',
#         'api_key': '<your Azure OpenAI API key here>',
#         'base_url': '<your Azure OpenAI API base here>',
#         'api_type': 'azure',
#         'api_version': '2024-02-15-preview',
#     },
# ]
# ```
# 
# You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

# %% [markdown]
# ## Example Task: Investment suggestion with realtime data
# 
# We consider a scenario where news data are streamed from a source, and we use an assistant agent to provide investment suggestions based on the data continually.
# 
# First, we use the following code to simulate the data stream process.

# %%
def get_market_news(ind, ind_upper):
    import json

    import requests

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    # url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&sort=LATEST&limit=5&apikey=demo'
    # r = requests.get(url)
    # data = r.json()
    # with open('market_news_local.json', 'r') as file:
    #     # Load JSON data from file
    #     data = json.load(file)
    data = {
        "feed": [
            {
                "title": "Palantir CEO Says Our Generation's Atomic Bomb Could Be AI Weapon - And Arrive Sooner Than You Think - Palantir Technologies  ( NYSE:PLTR ) ",
                "summary": "Christopher Nolan's blockbuster movie \"Oppenheimer\" has reignited the public discourse surrounding the United States' use of an atomic bomb on Japan at the end of World War II.",
                "overall_sentiment_score": 0.009687,
            },
            {
                "title": '3 "Hedge Fund Hotels" Pulling into Support',
                "summary": "Institutional quality stocks have several benefits including high-liquidity, low beta, and a long runway. Strategist Andrew Rocco breaks down what investors should look for and pitches 3 ideas.",
                "banner_image": "https://staticx-tuner.zacks.com/images/articles/main/92/87.jpg",
                "overall_sentiment_score": 0.219747,
            },
            {
                "title": "PDFgear, Bringing a Completely-Free PDF Text Editing Feature",
                "summary": "LOS ANGELES, July 26, 2023 /PRNewswire/ -- PDFgear, a leading provider of PDF solutions, announced a piece of exciting news for everyone who works extensively with PDF documents.",
                "overall_sentiment_score": 0.360071,
            },
            {
                "title": "Researchers Pitch 'Immunizing' Images Against Deepfake Manipulation",
                "summary": "A team at MIT says injecting tiny disruptive bits of code can cause distorted deepfake images.",
                "overall_sentiment_score": -0.026894,
            },
            {
                "title": "Nvidia wins again - plus two more takeaways from this week's mega-cap earnings",
                "summary": "We made some key conclusions combing through quarterly results for Microsoft and Alphabet and listening to their conference calls with investors.",
                "overall_sentiment_score": 0.235177,
            },
        ]
    }
    feeds = data["feed"][ind:ind_upper]
    feeds_summary = "\n".join(
        [
            f"News summary: {f['title']}. {f['summary']} overall_sentiment_score: {f['overall_sentiment_score']}"
            for f in feeds
        ]
    )
    return feeds_summary


data = asyncio.Future()


async def add_stock_price_data():
    # simulating the data stream
    for i in range(0, 5, 1):
        latest_news = get_market_news(i, i + 1)
        if data.done():
            data.result().append(latest_news)
        else:
            data.set_result([latest_news])
        # print(data.result())
        await asyncio.sleep(5)


data_task = asyncio.create_task(add_stock_price_data())

# %% [markdown]
# Then, we construct agents. An assistant agent is created to answer the question using LLM. A UserProxyAgent is created to ask questions, and add the new data in the conversation when they are available.

# %%
# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "timeout": 600,
        "cache_seed": 41,
        "config_list": config_list,
        "temperature": 0,
    },
    system_message="You are a financial expert.",
)
# create a UserProxyAgent instance named "user"
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
    default_auto_reply=None,
)


async def add_data_reply(recipient, messages, sender, config):
    await asyncio.sleep(0.1)
    data = config["news_stream"]
    if data.done():
        result = data.result()
        if result:
            news_str = "\n".join(result)
            result.clear()
            return (
                True,
                f"Just got some latest market news. Merge your new suggestion with previous ones.\n{news_str}",
            )
        return False, None


user_proxy.register_reply(autogen.AssistantAgent, add_data_reply, position=2, config={"news_stream": data})

# %% [markdown]
# We invoke the `a_initiate_chat()` method of the user proxy agent to start the conversation.

# %%
await user_proxy.a_initiate_chat(  # noqa: F704
    assistant,
    message="""Give me investment suggestion in 3 bullet points.""",
)
while not data_task.done() and not data_task.cancelled():
    reply = await user_proxy.a_generate_reply(sender=assistant)  # noqa: F704
    if reply is not None:
        await user_proxy.a_send(reply, assistant)  # noqa: F704


