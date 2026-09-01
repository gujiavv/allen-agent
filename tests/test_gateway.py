"""Gateway 层测试。全部脱网——用假 provider 模拟各种故障。"""
import time

import pytest

from gateway import Gateway, Provider
from gateway import policies as P
from gateway.core import AllProvidersFailed, RateLimited


# ---------- 造假 provider ----------
class _Usage:
    def __init__(self, pt, ct):
        self.prompt_tokens, self.completion_tokens = pt, ct


class _Resp:
    def __init__(self, text, pt=10, ct=20):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
        self.usage = _Usage(pt, ct)


class _Boom(Exception):
    """可重试的假故障：类名要在 RETRYABLE_NAMES 里才算瞬时故障。"""


class APITimeoutError(Exception):
    pass


def _provider(name, key="k"):
    return Provider(name=name, base_url="https://x.invalid/v1", api_key=key,
                    chat_model=f"{name}-model", embedding_model=f"{name}-embed",
                    price_in_per_1m=1.0, price_out_per_1m=2.0)


def _gateway(providers, **kw):
    kw.setdefault("cache_ttl", 0)   # 默认关缓存，避免用例间互相污染
    return Gateway(providers=providers, **kw)


def _stub(gw, provider_name, fn):
    """把某个 provider 的 chat.completions.create 换成 fn。"""
    p = next(p for p in gw.providers if p.name == provider_name)
    p._client = type("C", (), {
        "chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(fn)})()})()
    })()


# ---------- 错误分类 ----------
def test_401之类的错误不该重试():
    """key 错了重试多少次都是一样的结果，只会拖慢用户、放大账单。"""
    e = Exception(); e.status_code = 401
    assert P.is_retryable(e) is False


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 408])
def test_瞬时故障应该重试(code):
    e = Exception(); e.status_code = code
    assert P.is_retryable(e) is True


def test_超时类异常按类名识别为可重试():
    assert P.is_retryable(APITimeoutError("timeout")) is True


def test_429退避得比普通错误更久():
    """供应商已经在喊太快了，此时退避不够久等于火上浇油。"""
    normal = [P.backoff_delay(1, rate_limited=False) for _ in range(50)]
    limited = [P.backoff_delay(1, rate_limited=True) for _ in range(50)]
    assert max(normal) < min(limited)


def test_退避带抖动():
    """固定间隔重试会让并发失败的请求同时再次撞上供应商，形成重试风暴。"""
    delays = {P.backoff_delay(2) for _ in range(50)}
    assert len(delays) > 40, "退避应当带随机抖动，不能是固定值"


# ---------- 重试与故障转移 ----------
def test_可重试的错误会重试并最终成功():
    gw = _gateway([_provider("a")], max_attempts=3)
    calls = {"n": 0}

    def _create(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise APITimeoutError("网络抖动")
        return _Resp("终于成功了")

    _stub(gw, "a", _create)
    assert gw.chat([{"role": "user", "content": "hi"}]) == "终于成功了"
    assert calls["n"] == 3


def test_不可重试的错误立刻放弃():
    gw = _gateway([_provider("a")], max_attempts=5)
    calls = {"n": 0}

    def _create(**kw):
        calls["n"] += 1
        e = Exception("key 无效"); e.status_code = 401
        raise e

    _stub(gw, "a", _create)
    with pytest.raises(Exception):
        gw.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1, "401 只该尝试一次"


def test_主provider挂了自动转移到备用():
    gw = _gateway([_provider("主"), _provider("备")], max_attempts=1)

    def _fail(**kw):
        raise APITimeoutError("主挂了")

    _stub(gw, "主", _fail)
    _stub(gw, "备", lambda **kw: _Resp("备用回答"))

    assert gw.chat([{"role": "user", "content": "hi"}]) == "备用回答"
    # 计费要记在真正干活的那个 provider 头上
    assert gw.meter.by_provider["备"]["calls"] == 1
    assert "主" not in gw.meter.by_provider


def test_全部provider失败时抛出带原因的异常():
    gw = _gateway([_provider("a"), _provider("b")], max_attempts=1)
    for n in ("a", "b"):
        _stub(gw, n, lambda **kw: (_ for _ in ()).throw(APITimeoutError(f"{n} 挂了")))

    with pytest.raises(AllProvidersFailed) as ei:
        gw.chat([{"role": "user", "content": "hi"}])
    assert set(ei.value.errors) == {"a", "b"}, "异常里要带上每家的失败原因"


# ---------- 计费 ----------
def test_记录token与成本():
    gw = _gateway([_provider("a")])
    _stub(gw, "a", lambda **kw: _Resp("回答", pt=1000, ct=2000))
    gw.chat([{"role": "user", "content": "hi"}])

    u = gw.meter.snapshot()
    assert u["prompt_tokens"] == 1000 and u["completion_tokens"] == 2000
    # 单价 1.0/2.0 每百万 → 1000/1e6*1 + 2000/1e6*2 = 0.005
    assert u["estimated_cost"] == pytest.approx(0.005)


def test_usage缺失不影响主链路():
    """不是所有供应商都返回 usage。计费不准可以接受，请求失败不行。"""
    gw = _gateway([_provider("a")])

    class _NoUsage:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    _stub(gw, "a", lambda **kw: _NoUsage())
    assert gw.chat([{"role": "user", "content": "hi"}]) == "ok"


# ---------- 缓存 ----------
def test_相同请求命中缓存不再调用模型():
    gw = _gateway([_provider("a")], cache_ttl=60)
    calls = {"n": 0}

    def _create(**kw):
        calls["n"] += 1
        return _Resp("回答")

    _stub(gw, "a", _create)
    msgs = [{"role": "user", "content": "同一个问题"}]
    assert gw.chat(msgs) == gw.chat(msgs) == "回答"
    assert calls["n"] == 1, "第二次应当命中缓存"
    assert gw.cache.stats()["hits"] == 1


def test_不同请求不会串味():
    gw = _gateway([_provider("a")], cache_ttl=60)
    _stub(gw, "a", lambda **kw: _Resp(kw["messages"][0]["content"] + "-答"))
    assert gw.chat([{"role": "user", "content": "问题甲"}]) == "问题甲-答"
    assert gw.chat([{"role": "user", "content": "问题乙"}]) == "问题乙-答"


def test_缓存过期后重新调用():
    gw = _gateway([_provider("a")], cache_ttl=0.05)
    calls = {"n": 0}
    _stub(gw, "a", lambda **kw: (calls.__setitem__("n", calls["n"] + 1), _Resp("x"))[1])
    msgs = [{"role": "user", "content": "q"}]
    gw.chat(msgs)
    time.sleep(0.1)
    gw.chat(msgs)
    assert calls["n"] == 2


# ---------- 限流 ----------
def test_超出配额被拒绝():
    gw = _gateway([_provider("a")], rate_per_min=60)
    _stub(gw, "a", lambda **kw: _Resp("ok"))
    # 桶容量 60，一口气打空
    for _ in range(60):
        gw.chat([{"role": "user", "content": "q"}], caller="ip-1", use_cache=False)
    with pytest.raises(RateLimited):
        gw.chat([{"role": "user", "content": "q"}], caller="ip-1", use_cache=False)


def test_不同调用方各算各的():
    gw = _gateway([_provider("a")], rate_per_min=1)
    _stub(gw, "a", lambda **kw: _Resp("ok"))
    gw.chat([{"role": "user", "content": "q"}], caller="ip-1", use_cache=False)
    # ip-2 不该受 ip-1 影响
    gw.chat([{"role": "user", "content": "q"}], caller="ip-2", use_cache=False)


def test_令牌桶随时间补充():
    bucket = P.TokenBucket(rate_per_min=6000, burst=1)   # 每秒补 100 个
    assert bucket.acquire() is True
    assert bucket.acquire() is False
    time.sleep(0.05)
    assert bucket.acquire() is True


def test_未配置限流时不拦截():
    gw = _gateway([_provider("a")], rate_per_min=0)
    _stub(gw, "a", lambda **kw: _Resp("ok"))
    for _ in range(50):
        gw.chat([{"role": "user", "content": "q"}], caller="ip", use_cache=False)


# ---------- provider 注册 ----------
def test_没配key的provider不参与调度():
    """备用没配不该报错，只是不参与转移。"""
    gw = _gateway([_provider("a"), _provider("空", key="")])
    assert [p.name for p in gw.providers] == ["a", "空"]  # 显式传入时不过滤
    from gateway.providers import Provider as Pv
    assert Pv(name="x", base_url="u", api_key="", chat_model="m").usable is False


def test_没有任何可用provider时启动就报错():
    with pytest.raises(RuntimeError):
        Gateway(providers=[])
