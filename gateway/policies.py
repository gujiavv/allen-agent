# gateway/policies.py
"""横切策略：错误分类、重试退避、限流、缓存。

这些逻辑和「调哪个模型」无关，所以单独一层，可以独立测试。
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import OrderedDict, deque


# --------------------------------------------------------------------------
# 错误分类
# --------------------------------------------------------------------------
# Gateway 里最容易做错的一件事：无差别重试。
#
# 401（key 错了）、400（参数不合法）这类错误重试多少次都是同样的结果，
# 只会白白拉长用户等待、放大账单。而 429（限流）、5xx、超时是瞬时故障，
# 重试大概率能成。
#
# 更要紧的是 429：供应商已经在喊「你太快了」，此时无脑重试等于火上浇油，
# 会把自己的配额彻底打死。所以 429 要重试，但必须退避得比别的错误更久。

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
# 这些异常类名一律视为瞬时故障。用名字匹配而不是 import 具体异常类，
# 是为了不把 gateway 绑死在某个 SDK 的版本上。
RETRYABLE_NAMES = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "RateLimitError", "ConnectError", "ReadTimeout", "ConnectTimeout",
    "RemoteProtocolError", "ReadError",
}


def is_retryable(exc: Exception) -> bool:
    """判断一个异常值不值得重试。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    return type(exc).__name__ in RETRYABLE_NAMES


def is_rate_limit(exc: Exception) -> bool:
    """429 要退避得更久，单独识别。"""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return status == 429 or type(exc).__name__ == "RateLimitError"


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 8.0,
                  rate_limited: bool = False) -> float:
    """指数退避 + 抖动。attempt 从 0 开始。

    为什么要抖动（jitter）：如果十个并发请求同时失败、又同时按固定间隔重试，
    它们会在同一时刻再次撞上供应商，形成「重试风暴」，反而更难恢复。
    加一个随机量把它们错开。

    429 的基数翻倍：供应商已经在限流，退得比平时更久才有意义。
    """
    delay = min(base * (2 ** attempt), cap)
    if rate_limited:
        delay = min(delay * 2, cap * 2)
    return delay * (0.5 + random.random() * 0.5)  # 抖动到 50%~100%


# --------------------------------------------------------------------------
# 限流：令牌桶
# --------------------------------------------------------------------------
class TokenBucket:
    """令牌桶限流。

    为什么用令牌桶而不是「每分钟计数」的固定窗口：固定窗口有临界问题——
    限制每分钟 10 次时，用户可以在 0:59 发 10 次、1:01 再发 10 次，
    两秒内实际发了 20 次。令牌桶按时间连续补充，天然没有这个缺口，
    而且桶容量允许短时突发，比严格匀速对用户更友好。
    """

    def __init__(self, rate_per_min: float, burst: int | None = None):
        self.rate = rate_per_min / 60.0          # 每秒补充多少令牌
        self.capacity = burst if burst is not None else max(1, int(rate_per_min))
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> bool:
        """取 n 个令牌。取到返回 True，桶空返回 False（调用方决定拒绝还是等待）。"""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity, self._tokens + (now - self._last) * self.rate
            )
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class KeyedRateLimiter:
    """按调用方（IP、用户、API key）分别限流，互不影响。"""

    def __init__(self, rate_per_min: float, burst: int | None = None):
        self.rate_per_min, self.burst = rate_per_min, burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, n: int = 1) -> bool:
        if self.rate_per_min <= 0:      # 配 0 表示不限流
            return True
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = TokenBucket(self.rate_per_min, self.burst)
        return bucket.acquire(n)


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------
class ResponseCache:
    """按请求内容精确匹配的 LRU 缓存，带 TTL。

    刻意【不】做语义缓存（把问题向量化、找相似的旧问题直接返回）：
    那要为每次请求多付一次 embedding 调用和一次向量检索，而且「多相似算同一个
    问题」是个很难调的阈值——调松了会张冠李戴，返回上一个用户问题的答案，
    在客服场景里这是事故。精确匹配虽然命中率低，但绝不会答错。

    流式响应不进缓存：见 core.py 里的说明。
    """

    def __init__(self, maxsize: int = 256, ttl: float = 600.0):
        self.maxsize, self.ttl = maxsize, ttl
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def make_key(*parts) -> str:
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if time.monotonic() > expires_at:
                del self._data[key]         # 过期即删，不留垃圾
                self.misses += 1
                return None
            self._data.move_to_end(key)     # LRU：命中的挪到末尾
            self.hits += 1
            return value

    def put(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)   # 淘汰最久未使用的

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


# --------------------------------------------------------------------------
# 用量统计
# --------------------------------------------------------------------------
class Meter:
    """记录每次调用的 token 与成本，回答「今天花了多少钱」。

    只存在内存里，进程重启即清空。生产环境应该写进时序库或日志系统，
    但那属于可观测性基础设施，不是 Gateway 的核心职责。
    """

    def __init__(self, keep_recent: int = 200):
        self._lock = threading.Lock()
        self._recent = deque(maxlen=keep_recent)
        self.calls = self.failures = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.cost = 0.0
        self.by_provider: dict[str, dict] = {}

    def record(self, provider: str, model: str, prompt_tokens: int,
               completion_tokens: int, cost: float, latency_ms: float,
               cached: bool = False, ok: bool = True) -> None:
        with self._lock:
            self.calls += 1
            if not ok:
                self.failures += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.cost += cost

            slot = self.by_provider.setdefault(
                provider, {"calls": 0, "failures": 0, "prompt_tokens": 0,
                           "completion_tokens": 0, "cost": 0.0})
            slot["calls"] += 1
            slot["failures"] += 0 if ok else 1
            slot["prompt_tokens"] += prompt_tokens
            slot["completion_tokens"] += completion_tokens
            slot["cost"] += cost

            self._recent.append({
                "provider": provider, "model": model, "cached": cached, "ok": ok,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "cost": round(cost, 6), "latency_ms": round(latency_ms, 1),
            })

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "failures": self.failures,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "estimated_cost": round(self.cost, 6),
                "by_provider": {
                    k: {**v, "cost": round(v["cost"], 6)}
                    for k, v in self.by_provider.items()
                },
                "recent": list(self._recent)[-20:],
            }
