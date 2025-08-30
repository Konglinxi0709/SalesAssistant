import json
import os
from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv  # 新增：用于加载.env文件

# 加载.env文件中的环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY", "")
base_url = "https://api.deepseek.com/v1"


client = OpenAI(api_key=api_key, base_url=base_url)
def simple_call_llm(system_prompt, user_prompt, debug=False):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )
    content = response.choices[0].message.content
    if debug:
        print(content)
    return content


deepseek_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
async def call_reasoner(system_prompt, user_prompt, debug=False):
    response = await deepseek_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    reasoning = True
    result = ""
    if debug:
        print("【思考过程】：")
    async for chunk in response:
        if chunk.choices[0].delta.reasoning_content and debug:
            print(chunk.choices[0].delta.reasoning_content, end="")
        elif chunk.choices[0].delta.content:
            if debug:
                if reasoning:  # 思考结束
                    print("\n【输出】：")
                    reasoning = False
                print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
    return result