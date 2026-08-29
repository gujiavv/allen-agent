# llm.py
"""百炼大模型客户端。

trust_env=False: 本机全局代理(HTTPS_PROXY)会让到模型服务的 TLS 握手失败，
所以让 HTTP 客户端直连、忽略环境变量里的代理。若你的网络必须走代理，改成 True。
"""
import httpx
from openai import OpenAI

import config

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
    http_client=httpx.Client(trust_env=False, timeout=60.0),
)


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """调用大模型并取出回复文本。异常由调用方处理。"""
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
