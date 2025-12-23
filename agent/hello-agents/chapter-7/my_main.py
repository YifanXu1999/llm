# my_main.py
from dotenv import load_dotenv
from my_llm import MyLLM
import os
# 加载环境变量
load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL_ID")
# 实例化我们重写的客户端，并指定provider
llm = MyLLM(provider="DeepSeek", api_key=API_KEY, base_url=BASE_URL, model=MODEL_ID)

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm.think(messages)

# 打印响应
print("DeepSeek Response:")
for chunk in response_stream:
    # chunk 已经是文本片段，可以直接使用
    print(chunk, end="", flush=True)