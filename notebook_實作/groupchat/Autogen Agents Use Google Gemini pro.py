"""
Autogen 與開放式 AI 模型配合得很好，但 Gemini pro 與 auto gen 的整合尚未完成。但我沒有耐心等待:)，發現工作幾乎完成了，但還沒有合併到 main 中，這裡是如何將 Gemini Pro 與 Microsoft Autogen 一起使用的 TL;DR
注意：目前還沒有這方面的官方文檔，預計在不久的將來
mkdir ~/autogen-demo
~/autogen-demo
pip install git+https://github.com/microsoft/autogen.git@gemini

建立 OAI_CONFIG_LIST 文件
[
    {
        "model": "gemini-pro",
        "api_key": "your-gemini-pro-key",
        "api_type": "google"
    }
]

 建立一個 demo.py 文件
 https://medium.com/@zelarsoft/using-google-gemini-pro-with-autogen-agents-08b55bba69db
 運行會報錯 ^_^
"""
import autogen
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent
from autogen.code_utils import DEFAULT_MODEL, UNKNOWN, content_str, execute_code, extract_code, infer_lang
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo-1106"],
    },
)
assistant = AssistantAgent(
    "assistant", llm_config={"config_list": config_list, "seed": 42}, max_consecutive_auto_reply=4
)
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
)
user_proxy.initiate_chat(assistant, message="""
    Plot a chart of NVDA and TESLA stock price change YTD, Save the result to a file named nvda_tesla.png
""")
