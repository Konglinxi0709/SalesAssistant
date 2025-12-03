import os
import asyncio
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI, AsyncOpenAI
import socket  # 需要引入 socket
import httpx   # 需要确保安装了 httpx

# 加载环境变量
load_dotenv()

# 配置 Azure 信息
# 注意：实际部署名称(deployment)可能与模型名称不同，请根据 Azure 门户中的“Deployments”填写
AZURE_ENDPOINT = "https://admin-meau96fd-eastus2.cognitiveservices.azure.com/"
API_KEY = os.getenv("AZURE_API_KEY", "")
API_VERSION = "2024-12-01-preview" 

# 定义模型对应的 Deployment Name (请根据你的 Azure 后台实际部署名修改)
CHAT_DEPLOYMENT = "gpt-5-mini"      
REASONER_DEPLOYMENT = "DeepSeek-R1-0528"  


class KeepAliveTransport(httpx.AsyncHTTPTransport):
    async def _connect(self, *args, **kwargs):
        # 调用父类建立连接，获取 stream
        stream = await super()._connect(*args, **kwargs)
        
        # 获取底层 socket 对象
        # 注意：不同 httpx 版本的内部结构可能不同，这里针对常见版本
        if hasattr(stream, "socket"):
            sock = stream.socket
        elif hasattr(stream, "stream") and hasattr(stream.stream, "socket"):
             # 针对 httpcore 的层级结构
            sock = stream.stream.socket
        else:
            # 如果找不到 socket，为了防止报错，直接返回
            return stream

        if sock:
            # 开启 TCP KeepAlive 机制
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # 下面设置因操作系统而异 (Linux/Mac vs Windows)
            # 目的是：空闲 60秒后开始探测，每 15秒发一次，失败 3次才算断开
            
            if hasattr(socket, 'TCP_KEEPIDLE'): # Linux / Mac
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            
        return stream

# 1. 创建自定义的 HTTP Client
http_client = httpx.AsyncClient(
    transport=KeepAliveTransport(),
    # 这里设置 httpx 的读取超时，必须大于你的预估推理时间
    timeout=httpx.Timeout(600.0, connect=60.0, read=600.0, write=60.0, pool=60.0)
)

# 1. 初始化同步客户端 (用于普通调用)
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    timeout=600,
)

# 2. 初始化异步客户端 (用于异步和流式调用)
async_client = AsyncAzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    timeout=600,
    http_client=http_client
)

def simple_call_llm(system_prompt, user_prompt, temperature=1.0, debug=False):
    """
    同步非流式调用 (对应 gpt-4o 等快速模型)
    """
    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False,
        temperature=temperature,
        timeout=600
    )
    content = response.choices[0].message.content
    if debug:
        print(f"【Debug Output】: {content}")
    return content

async def call_reasoner(system_prompt, user_prompt, temperature=1.0, reasoning_effort="low", debug=False):
    """
    异步流式调用，尝试捕获推理过程。
    注意：Azure/OpenAI 的 o1/o3 系列模型目前可能隐藏具体的 reasoning_content 文本，
    如果 API 不返回该字段，本函数会自动降级只输出最终结果。
    """
    
    response = await async_client.chat.completions.create(
        model=REASONER_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=600
    )
    
    reasoning_result = ""
    result = ""
    is_collecting_reasoning = False
    if debug:
        print("【开始调用推理模型】...")
    async for chunk in response:
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        
        # 尝试获取推理内容 (DeepSeek 风格)
        # 注意：OpenAI Python SDK 标准对象可能没有 reasoning_content 属性，需用 getattr 安全获取
        r_content = getattr(delta, 'reasoning_content', None)
        content = delta.content
        # 处理推理部分
        if r_content:
            if debug:
                print(r_content, end="", flush=True)
            reasoning_result += r_content
        
        # 处理最终结果部分
        if content:
            # 如果之前在推理，现在有了 content，说明推理结束（仅针对视觉调试）
            if debug and not result and reasoning_result:
                print("\n【推理结束，输出结果】：")
            
            if debug:
                print(content, end="", flush=True)
            result += content
    if debug:
        print() # 换行
    return reasoning_result, result


async def call_chat(system_prompt, user_prompt, temperature=1.0, debug=False):
    """
    异步流式调用 (普通对话模型)
    """
    response = await async_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=temperature,
        timeout=600
    )
    
    result = ""
    if debug:
        print("【Chat输出】：")
        
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if debug:
                print(content, end="", flush=True)
            result += content
    
    if debug:
        print()
    return result
        
