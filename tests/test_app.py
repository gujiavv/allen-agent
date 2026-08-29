"""allen-agent 接口测试。"""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_myapp(client):
    r = client.get("/myapp")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "body": {"name": "张三丰"}}


def test_chat_success(client, mock_llm):
    mock_llm("你好，我是测试回复")
    r = client.post("/chat", json={"message": "你好"})
    assert r.status_code == 200
    assert r.json()["reply"] == "你好，我是测试回复"


def test_chat_passes_model_and_message(client, mock_llm):
    """确认请求体里的 message 和环境变量里的 model 被正确传给大模型。"""
    setter = mock_llm()
    client.post("/chat", json={"message": "hello world"})
    kwargs = setter.called_with
    assert kwargs["model"] == "qwen-test-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hello world"}]


def test_chat_upstream_error_returns_500(client, monkeypatch):
    """大模型报错时应转成 500，而不是把异常直接抛出去。"""
    import llm

    def _boom(*args, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(llm.client.chat.completions, "create", _boom)
    r = client.post("/chat", json={"message": "x"})
    assert r.status_code == 500
    assert "upstream exploded" in r.json()["detail"]


def test_chat_rejects_malformed_body(client):
    """缺少 message 字段应被 pydantic 拦下，返回 422。"""
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"msg": "wrong field"}).status_code == 422


def test_ui_is_mounted(client):
    """Gradio 页面挂在 /ui（Gradio 会把 /ui 重定向到 /ui/）。"""
    r = client.get("/ui", follow_redirects=True)
    assert r.status_code == 200
