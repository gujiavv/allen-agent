"""流式接口 /chat/stream 的测试。全部脱网。"""
import json


def _parse_sse(text):
    """把 SSE 文本拆成 [(type, value)]。"""
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            e = json.loads(line[6:])
            out.append((e["type"], e["value"]))
    return out


def test_流式返回完整事件序列(client, mock_llm_stream, fake_retrieval):
    mock_llm_stream(thinking="思考", content="回答")
    fake_retrieval((0.90, "命中文章A"))

    events = _parse_sse(client.post("/chat/stream", json={"message": "问题"}).text)
    types = [t for t, _ in events]

    # mode 和 sources 必须排在正文之前，前端才能先把引用显示出来
    assert types[0] == "mode"
    assert types[1] == "sources"
    assert types[-1] == "done"
    assert "thinking" in types and "content" in types


def test_流式_域内问题走rag并先推送引用(client, mock_llm_stream, fake_retrieval):
    mock_llm_stream()
    fake_retrieval((0.90, "命中文章A"), (0.60, "命中文章B"))

    events = dict(_parse_sse(client.post("/chat/stream", json={"message": "问"}).text))
    assert events["mode"] == "rag"
    assert [s["title"] for s in events["sources"]] == ["命中文章A", "命中文章B"]


def test_流式_域外问题退回大模型且无引用(client, mock_llm_stream, fake_retrieval):
    mock_llm_stream()
    fake_retrieval((0.30, "不相关"))

    events = dict(_parse_sse(client.post("/chat/stream", json={"message": "问"}).text))
    assert events["mode"] == "llm"
    assert events["sources"] == []


def test_流式_增量拼起来等于完整文本(client, mock_llm_stream, fake_retrieval):
    mock_llm_stream(thinking="先想一下", content="然后回答")
    fake_retrieval((0.90, "文章"))

    events = _parse_sse(client.post("/chat/stream", json={"message": "问"}).text)
    assert "".join(v for t, v in events if t == "thinking") == "先想一下"
    assert "".join(v for t, v in events if t == "content") == "然后回答"


def test_流式_检索失败时降级而非中断(client, mock_llm_stream, fake_retrieval):
    mock_llm_stream()
    fake_retrieval(raises=True)

    r = client.post("/chat/stream", json={"message": "问"})
    assert r.status_code == 200
    events = dict(_parse_sse(r.text))
    assert events["mode"] == "llm"


def test_流式_大模型报错时发error事件而不是断流(client, monkeypatch, fake_retrieval):
    """流已经开始发送后再报错，没法改 HTTP 状态码，只能用 error 事件告知前端。"""
    import llm

    def _boom(messages, temperature=0.7):
        raise RuntimeError("模型炸了")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "chat_stream", _boom)
    fake_retrieval((0.90, "文章"))

    events = dict(_parse_sse(client.post("/chat/stream", json={"message": "问"}).text))
    assert "模型炸了" in events["error"]
    assert "done" in [t for t, _ in _parse_sse(client.post("/chat/stream", json={"message": "问"}).text)]


def test_非流式chat接口不受影响(client, mock_llm, fake_retrieval):
    """/chat 的契约必须保持原样，外部调用方不该被流式改造波及。"""
    mock_llm("完整回答")
    fake_retrieval((0.90, "文章"))

    body = client.post("/chat", json={"message": "问"}).json()
    assert body["reply"] == "完整回答"
    assert body["mode"] == "rag"
    assert len(body["sources"]) == 1
