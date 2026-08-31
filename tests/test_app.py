"""allen-agent 接口测试。"""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["commit"]        # 有 Railway 变量用它，否则读本地 .git


def test_health_报告rag状态(client, fake_retrieval, mock_llm):
    """/health 要能反映向量库到底加载没加载。

    索引没进镜像时服务照样能起、health 照样 ok，RAG 却是死的。
    这个字段就是用来戳穿那种静默降级的。
    """
    # client 夹具默认没有向量库
    assert client.get("/health").json()["rag_enabled"] is False

    fake_retrieval((0.9, "文章"))
    assert client.get("/health").json()["rag_enabled"] is True


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


def test_health_即使统计失败也不崩(client, monkeypatch):
    """Railway 靠 /health 判断存活，它抛异常会让整个部署被判失败。"""
    from rag import pipeline

    class _Broken:
        @property
        def _collection(self):
            raise RuntimeError("Chroma 内部结构变了")

    monkeypatch.setattr(pipeline, "_store", _Broken())
    monkeypatch.setattr(pipeline, "_loaded", True)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
