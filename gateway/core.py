# gateway/core.py
"""Gateway 主体：把 provider、重试、故障转移、缓存、计费编排起来。

对外只暴露 chat / chat_stream / embed 三个方法，调用方不需要知道
背后有几个 provider、重试了几次、命中没命中缓存。
"""
from __future__ import annotations

import logging
import time

from gateway import policies as P
from gateway.providers import Provider, build_registry

logger = logging.getLogger(__name__)


class AllProvidersFailed(Exception):
    """所有 provider 都失败了。带上每家的失败原因，便于排查。"""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__(
            "所有供应商均调用失败：" +
            "；".join(f"{k}({v})" for k, v in errors.items())
        )


class RateLimited(Exception):
    """调用方超出配额。这是 Gateway 主动拒绝，不是供应商拒绝。"""


class Gateway:
    def __init__(self, providers: list[Provider] | None = None,
                 max_attempts: int = 3, rate_per_min: float = 0,
                 cache_ttl: float = 600.0, cache_size: int = 256):
        self.providers = providers if providers is not None else build_registry()
        if not self.providers:
            raise RuntimeError(
                "没有可用的 provider。至少要配置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL。"
            )
        # 每个 provider 内部重试的次数。注意总尝试次数是
        # max_attempts × provider 数量，别把两者都调大。
        self.max_attempts = max_attempts
        self.limiter = P.KeyedRateLimiter(rate_per_min)
        self.cache = P.ResponseCache(maxsize=cache_size, ttl=cache_ttl)
        self.meter = P.Meter()

    # ------------------------------------------------------------------
    def _call_with_retry(self, provider: Provider, fn, *args, **kwargs):
        """在单个 provider 上带重试地执行一次调用。

        不可重试的错误立刻抛出——继续重试 401 只是在浪费时间。
        """
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn(provider, *args, **kwargs)
            except Exception as e:
                last = e
                if not P.is_retryable(e):
                    logger.warning("[%s] 不可重试的错误：%s", provider.name, e)
                    raise
                if attempt == self.max_attempts - 1:
                    break
                delay = P.backoff_delay(attempt, rate_limited=P.is_rate_limit(e))
                logger.warning("[%s] 第 %d 次失败（%s），%.1fs 后重试",
                               provider.name, attempt + 1, type(e).__name__, delay)
                time.sleep(delay)
        raise last  # type: ignore[misc]

    def _failover(self, fn, *args, **kwargs):
        """按注册顺序逐个 provider 尝试，全失败才抛 AllProvidersFailed。"""
        errors: dict[str, str] = {}
        for provider in self.providers:
            try:
                return provider, self._call_with_retry(provider, fn, *args, **kwargs)
            except Exception as e:
                errors[provider.name] = f"{type(e).__name__}: {e}"
                logger.warning("[%s] 已耗尽重试，转移到下一个 provider", provider.name)
        raise AllProvidersFailed(errors)

    def _check_rate(self, caller: str | None) -> None:
        if caller and not self.limiter.allow(caller):
            raise RateLimited(f"调用方 {caller} 已超出配额，请稍后再试")

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.7,
             caller: str | None = None, use_cache: bool = True) -> str:
        """非流式对话。返回回复文本。"""
        self._check_rate(caller)

        key = P.ResponseCache.make_key("chat", messages, temperature)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                # 缓存命中不产生 token 消耗，但要记一笔，
                # 否则统计里看不出缓存到底省了多少
                self.meter.record("cache", "-", 0, 0, 0.0, 0.0, cached=True)
                return cached

        def _do(provider: Provider):
            return provider.client.chat.completions.create(
                model=provider.chat_model, messages=messages, temperature=temperature,
            )

        t0 = time.perf_counter()
        try:
            provider, response = self._failover(_do)
        except AllProvidersFailed:
            self.meter.record("-", "-", 0, 0, 0.0,
                              (time.perf_counter() - t0) * 1000, ok=False)
            raise

        # usage 不是所有供应商都返回。计费是辅助功能，字段缺失只该让统计不准，
        # 绝不能让主链路失败——用户要的是回答，不是账单。
        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.meter.record(provider.name, provider.chat_model, pt, ct,
                          provider.cost(pt, ct), (time.perf_counter() - t0) * 1000)

        text = response.choices[0].message.content
        if use_cache:
            self.cache.put(key, text)
        return text

    # ------------------------------------------------------------------
    def chat_stream(self, messages: list[dict], temperature: float = 0.7,
                    caller: str | None = None):
        """流式对话，逐段产出 (类型, 文本)，类型为 thinking / content。

        ⚠️ 流式下故障转移只在【第一个分片到达之前】有效。

        一旦有内容发给了客户端，就不能再切换 provider 重来——用户已经看到
        半句话了，换个模型重新生成会导致回答从中间断裂、前后矛盾。所以这里
        把「建立流」和「消费流」分开：建流阶段允许失败转移，进入消费阶段后
        任何异常都直接抛给调用方，由上层决定怎么向用户交代。

        流式也不走缓存：要缓存就得先把整个流收完再回放，那就失去了流式
        「首字快」的意义。
        """
        self._check_rate(caller)

        def _open_stream(provider: Provider):
            return provider.client.chat.completions.create(
                model=provider.chat_model, messages=messages,
                temperature=temperature, stream=True,
            )

        t0 = time.perf_counter()
        # 建流阶段：这一步失败可以安全地换 provider，因为还没有任何内容发出去
        provider, stream = self._failover(_open_stream)

        pt = ct = 0
        try:
            # 消费阶段：从这里开始不再转移
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage:  # 有些供应商在最后一个分片里带 usage
                    pt = getattr(usage, "prompt_tokens", 0) or pt
                    ct = getattr(usage, "completion_tokens", 0) or ct
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # reasoning_content 非 OpenAI 官方字段，用 getattr 取
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield "thinking", reasoning
                if delta.content:
                    yield "content", delta.content
        except Exception:
            self.meter.record(provider.name, provider.chat_model, pt, ct, 0.0,
                              (time.perf_counter() - t0) * 1000, ok=False)
            raise
        else:
            self.meter.record(provider.name, provider.chat_model, pt, ct,
                              provider.cost(pt, ct),
                              (time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    def embed(self, texts: list[str], caller: str | None = None) -> list[list[float]]:
        """批量向量化。

        只在声明了 embedding_model 的 provider 之间转移——DeepSeek 这类
        没有 embedding 接口的供应商，做对话的备份可以，接管检索不行。
        """
        self._check_rate(caller)
        capable = [p for p in self.providers if p.embedding_model]
        if not capable:
            raise RuntimeError("没有任何 provider 提供 embedding 能力")

        def _do(provider: Provider):
            return provider.client.embeddings.create(
                model=provider.embedding_model, input=texts,
            )

        errors: dict[str, str] = {}
        t0 = time.perf_counter()
        for provider in capable:
            try:
                response = self._call_with_retry(provider, _do)
            except Exception as e:
                errors[provider.name] = f"{type(e).__name__}: {e}"
                continue
            pt = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
            self.meter.record(provider.name, provider.embedding_model, pt, 0,
                              provider.cost(pt, 0),
                              (time.perf_counter() - t0) * 1000)
            return [d.embedding for d in response.data]
        raise AllProvidersFailed(errors)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "providers": [p.name for p in self.providers],
            "usage": self.meter.snapshot(),
            "cache": self.cache.stats(),
        }
