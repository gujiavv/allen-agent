"""访问密码测试。

conftest 默认不设 UI_PASSWORD，所以其余测试文件不受鉴权影响；
本文件按需 monkeypatch 打开鉴权。
"""
import pytest

import config


@pytest.fixture
def with_password(monkeypatch):
    monkeypatch.setattr(config, "UI_PASSWORD", "test-secret")
    return "test-secret"


def test_未配密码时接口完全开放(client, mock_llm):
    """本地开发不该被鉴权挡住。"""
    mock_llm("回答")
    assert client.post("/chat", json={"message": "问"}).status_code == 200


def test_配了密码后缺请求头返回401(client, with_password, mock_llm):
    mock_llm("回答")
    r = client.post("/chat", json={"message": "问"})
    assert r.status_code == 401
    assert "密码" in r.json()["detail"]


def test_密码错误返回401(client, with_password, mock_llm):
    mock_llm("回答")
    r = client.post("/chat", json={"message": "问"},
                    headers={"X-API-Password": "wrong"})
    assert r.status_code == 401


def test_密码正确放行(client, with_password, mock_llm):
    mock_llm("回答")
    r = client.post("/chat", json={"message": "问"},
                    headers={"X-API-Password": with_password})
    assert r.status_code == 200
    assert r.json()["reply"] == "回答"


def test_流式接口同样受保护(client, with_password, mock_llm_stream):
    """只锁 /chat 不锁 /chat/stream 等于没锁。"""
    mock_llm_stream()
    assert client.post("/chat/stream", json={"message": "问"}).status_code == 401
    r = client.post("/chat/stream", json={"message": "问"},
                    headers={"X-API-Password": with_password})
    assert r.status_code == 200


def test_health和myapp不需要密码(client, with_password):
    """Railway 靠 /health 做存活探测，加鉴权会让部署被判失败。"""
    assert client.get("/health").status_code == 200
    assert client.get("/myapp").status_code == 200


def test_health如实报告是否启用鉴权(client, monkeypatch):
    """忘了在部署平台配密码时，服务照常能用但接口是敞开的，得能查出来。"""
    monkeypatch.setattr(config, "UI_PASSWORD", "")
    assert client.get("/health").json()["auth_enabled"] is False

    monkeypatch.setattr(config, "UI_PASSWORD", "x")
    assert client.get("/health").json()["auth_enabled"] is True


def test_页面登录校验(with_password):
    import ui

    assert ui._check_login("随便填", "test-secret") is True
    assert ui._check_login("随便填", "wrong") is False


def test_未配密码时页面登录一律拒绝(monkeypatch):
    """避免空密码变成万能钥匙——没配密码时不该走登录流程。"""
    import ui

    monkeypatch.setattr(config, "UI_PASSWORD", "")
    assert ui._check_login("any", "") is False
    assert ui._check_login("any", "whatever") is False


def test_页面自调用会带上密码头(with_password, mock_llm_stream, fake_retrieval):
    """页面走的是同一个受保护的接口，不给内部调用开后门。"""
    import asyncio

    import ui

    mock_llm_stream(thinking="想", content="页面拿到的回答")
    fake_retrieval((0.90, "文章"))

    frames = asyncio.run(_collect(ui, "问题"))
    text = "\n".join(m.content for m in frames[-1])
    assert "页面拿到的回答" in text      # 没被 401 挡下
    assert "401" not in text


async def _collect(ui, message):
    frames = []
    async for f in ui._respond(message, []):
        frames.append(f)
    return frames
