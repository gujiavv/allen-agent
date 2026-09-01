# llm.py
"""模型调用入口。真正的逻辑在 gateway/ 里。

保留这一层薄封装而不是让 rag/pipeline.py 直接调 Gateway：
调用方的接口（llm.chat / llm.chat_stream）完全没变，底层从裸 SDK 换成
Gateway 是一次对上层透明的替换。测试里 monkeypatch llm.chat 的写法也照旧有效。
"""
import os

from gateway import Gateway

# 进程内单例：Gateway 持有连接池、缓存和计量器，每次请求新建会丢失全部状态。
gateway = Gateway(
    max_attempts=int(os.getenv("LLM_MAX_ATTEMPTS", "3")),
    # 默认 0 表示不限流。线上按需在环境变量里开，例如每个调用方每分钟 20 次。
    rate_per_min=float(os.getenv("LLM_RATE_PER_MIN", "0")),
    cache_ttl=float(os.getenv("LLM_CACHE_TTL", "600")),
)


def chat(messages: list[dict], temperature: float = 0.7,
         caller: str | None = None) -> str:
    """调用大模型并取出回复文本。异常由调用方处理。"""
    return gateway.chat(messages, temperature=temperature, caller=caller)


def chat_stream(messages: list[dict], temperature: float = 0.7,
                caller: str | None = None):
    """流式调用，逐段产出 ("thinking"|"content", 文本)。

    qwen3.7-plus 默认就会返回 reasoning_content，不需要额外传 enable_thinking。
    """
    yield from gateway.chat_stream(messages, temperature=temperature, caller=caller)


def stats() -> dict:
    """用量与缓存统计，供 /gateway/stats 使用。"""
    return gateway.stats()
