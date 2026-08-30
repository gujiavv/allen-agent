"""自动路由测试：什么时候走文档、什么时候退回大模型。全部脱网。"""


def test_分数高于阈值走rag并带来源(client, mock_llm, fake_retrieval):
    mock_llm("基于文档的回答")
    fake_retrieval((0.90, "命中文章A"), (0.60, "命中文章B"))
    body = client.post("/chat", json={"message": "Ballet 卡片是正品吗"}).json()

    assert body["mode"] == "rag"
    assert [s["title"] for s in body["sources"]] == ["命中文章A", "命中文章B"]
    assert body["sources"][0]["score"] == 0.9


def test_分数低于阈值退回大模型(client, mock_llm, fake_retrieval):
    mock_llm("模型自身知识的回答")
    fake_retrieval((0.30, "不相关文章"))
    body = client.post("/chat", json={"message": "今天天气怎么样"}).json()

    assert body["mode"] == "llm"
    assert body["sources"] == []


def test_只把过阈的块塞进上下文(client, mock_llm, fake_retrieval):
    """top1 过阈才走 RAG，但低分块不该混进上下文污染回答。"""
    mock_llm("回答")
    fake_retrieval((0.90, "高分文章"), (0.20, "低分文章"))
    body = client.post("/chat", json={"message": "问题"}).json()

    assert [s["title"] for s in body["sources"]] == ["高分文章"]


def test_rag模式下系统提示词包含检索到的资料(client, mock_llm, fake_retrieval):
    setter = mock_llm("回答")
    fake_retrieval((0.90, "命中文章A"))
    client.post("/chat", json={"message": "问题"})

    messages = setter.called_with["messages"]
    assert messages[0]["role"] == "system"
    assert "命中文章A" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "问题"}


def test_检索失败时降级为大模型而不是报500(client, mock_llm, fake_retrieval):
    """embedding 接口挂掉不该让整个请求失败。"""
    mock_llm("降级后的回答")
    fake_retrieval(raises=True)
    r = client.post("/chat", json={"message": "问题"})

    assert r.status_code == 200
    assert r.json()["mode"] == "llm"


def test_没有向量库时纯大模型模式(client, mock_llm):
    """client 夹具默认就没有向量库，此时不该调检索、也不该报错。"""
    mock_llm("纯模型回答")
    body = client.post("/chat", json={"message": "问题"}).json()

    assert body["mode"] == "llm"
    assert body["reply"] == "纯模型回答"


def test_页面自调用不依赖端口(monkeypatch, mock_llm_stream, fake_retrieval):
    """回归测试：Gradio 页面必须打到本应用自己的 /chat/stream。

    早先的实现是 POST 到 http://127.0.0.1:{PORT}/chat，靠 PORT 环境变量猜端口。
    实际端口由 uvicorn 的 --port 决定，两者不一致时自调用会打到别的服务上
    （曾实测打进了 8000 端口上的一条 SSH 隧道，返回了毫不相干的结果）。
    这里把 PORT 设成一个明显错误的值，页面链路仍须正常工作。
    """
    import asyncio

    import ui

    monkeypatch.setenv("PORT", "59999")  # 故意设错，不该有任何影响
    mock_llm_stream(thinking="先想想", content="基于文档的回答")
    fake_retrieval((0.90, "命中文章A"))

    text = asyncio.run(_collect_reply(ui, "Ballet 卡片是正品吗"))

    assert "基于文档的回答" in text
    assert "命中文章A" in text           # 引用被渲染出来 → 确实走了 rag
    assert "知识库中没有相关文档" not in text


def test_页面把思考过程渲染成折叠面板(mock_llm_stream, fake_retrieval):
    """思考过程要单独成一条带 metadata 的消息，Gradio 才会渲染成可折叠面板。"""
    import asyncio

    import ui

    mock_llm_stream(thinking="我的推理过程", content="最终回答")
    fake_retrieval((0.90, "文章"))

    frames = asyncio.run(_collect_frames(ui, "问题"))
    last = frames[-1]

    thinking_msgs = [m for m in last if m.metadata]
    assert len(thinking_msgs) == 1, "思考应当单独成一条消息"
    assert thinking_msgs[0].content == "我的推理过程"
    assert "思考" in thinking_msgs[0].metadata["title"]
    assert thinking_msgs[0].metadata["status"] == "done"  # 正文已到 → 思考结束

    answer_msgs = [m for m in last if not m.metadata]
    assert "最终回答" in answer_msgs[0].content


def test_页面流式逐帧递增(mock_llm_stream, fake_retrieval):
    """每个增量都该触发一次 yield，否则页面上看不到"逐字蹦出来"的效果。"""
    import asyncio

    import ui

    mock_llm_stream(thinking="abc", content="xyz")
    fake_retrieval((0.90, "文章"))

    frames = asyncio.run(_collect_frames(ui, "问题"))
    # 3 个思考字符 + 3 个正文字符，各触发一帧，加最后的收尾帧
    assert len(frames) >= 6


async def _collect_frames(ui, message):
    """把异步生成器产出的每一帧收集起来。"""
    frames = []
    async for frame in ui._respond(message, []):
        frames.append(frame)
    return frames


async def _collect_reply(ui, message):
    """取最后一帧并拼成纯文本，便于断言。"""
    frames = await _collect_frames(ui, message)
    last = frames[-1]
    if isinstance(last, str):
        return last
    return "\n".join(m.content for m in last)
