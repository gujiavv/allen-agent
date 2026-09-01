# gateway/providers.py
"""Provider（模型供应商）抽象。

Gateway 的第一件事就是把「供应商」变成数据而不是代码里写死的常量。
在此之前 llm.py 里 base_url / api_key / 模型名是硬编码的，换供应商要改代码；
之后换供应商只是换一条配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx
from openai import OpenAI


@dataclass
class Provider:
    """一个模型供应商的全部信息。

    定价单位是「每百万 token 的价格」，用于计费统计。
    ⚠️ 默认值是占位的 0.0——真实单价请查各家控制台后自己填，
    编一个价格出来比留空更糟糕。
    """

    name: str
    base_url: str
    api_key: str
    chat_model: str
    embedding_model: str | None = None

    price_in_per_1m: float = 0.0   # 输入（prompt）每百万 token 单价
    price_out_per_1m: float = 0.0  # 输出（completion）每百万 token 单价

    timeout: float = 60.0
    # 本机全局代理会让到模型服务的 TLS 握手失败，所以默认不走环境变量里的代理。
    # 若你的网络必须经代理才能出网，把这项设为 True。
    trust_env: bool = False

    _client: OpenAI | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> OpenAI:
        """懒加载并复用客户端。

        不在 __init__ 里建：注册表里可能配了好几个 provider，
        但一次请求通常只用到主 provider，没必要为备用的也建连接池。
        """
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=httpx.Client(
                    trust_env=self.trust_env, timeout=self.timeout
                ),
            )
        return self._client

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """估算单次调用的成本。单价没配时返回 0，不假装知道。"""
        return (
            prompt_tokens / 1_000_000 * self.price_in_per_1m
            + completion_tokens / 1_000_000 * self.price_out_per_1m
        )

    @property
    def usable(self) -> bool:
        """没有 key 的 provider 不参与调度，而不是等到调用时才报错。"""
        return bool(self.api_key and self.base_url)


def build_registry() -> list[Provider]:
    """从环境变量装配 provider 列表，顺序即优先级。

    第一个是主力，后面的是故障转移目标。只配了主 provider 也能跑——
    没有 key 的会被自动过滤掉，不会因为「备用没配」而报错。
    """
    candidates = [
        Provider(
            name="dashscope",
            base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            chat_model=os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus"),
            embedding_model=os.getenv(
                "DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding"
            ),
            price_in_per_1m=float(os.getenv("DASHSCOPE_PRICE_IN", "0") or 0),
            price_out_per_1m=float(os.getenv("DASHSCOPE_PRICE_OUT", "0") or 0),
        ),
        # 备用。只有配了 DEEPSEEK_API_KEY 才会启用。
        # 注意 DeepSeek 没有 embedding 接口，所以 embedding_model 留空——
        # 它只能作为对话的故障转移目标，不能接管检索。
        Provider(
            name="deepseek",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            chat_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            embedding_model=None,
            price_in_per_1m=float(os.getenv("DEEPSEEK_PRICE_IN", "0") or 0),
            price_out_per_1m=float(os.getenv("DEEPSEEK_PRICE_OUT", "0") or 0),
        ),
    ]
    return [p for p in candidates if p.usable]
