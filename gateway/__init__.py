# gateway/__init__.py
"""LLM Gateway：应用与模型供应商之间的一层。

职责：多 provider 抽象与故障转移、重试退避、限流、缓存、用量计费。
调用方只跟 gateway 打交道，不直接碰任何供应商 SDK。
"""
from gateway.core import AllProvidersFailed, Gateway, RateLimited
from gateway.providers import Provider, build_registry

__all__ = ["Gateway", "Provider", "build_registry",
           "AllProvidersFailed", "RateLimited"]
