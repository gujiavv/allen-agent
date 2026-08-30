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


def chat_stream(messages: list[dict], temperature: float = 0.7):
    """流式调用，逐段产出 (类型, 文本)。

    类型为 "thinking"（模型的推理过程）或 "content"（正式回答）。

    qwen3.7-plus 默认就会返回 reasoning_content，不需要额外传 enable_thinking。
    注意推理过程通常是英文的，即使提问和回答都是中文——这是模型行为，
    提示词左右不了它。
    """
    stream = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # reasoning_content 不是 OpenAI 官方字段，用 getattr 取，避免旧版 SDK 报错
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield "thinking", reasoning
        if delta.content:
            yield "content", delta.content
