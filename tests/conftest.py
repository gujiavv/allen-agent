"""测试夹具。

config.py 在 import 阶段就会校验 DASHSCOPE_API_KEY，所以必须在导入 app 之前
把环境变量准备好。所有夹具都不联网：大模型调用和向量检索都被替换掉。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 用 setdefault 先占位，这样 config.py 里的 load_dotenv 不会覆盖它们
# （python-dotenv 默认不覆盖已存在的环境变量）
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-key-for-ci")
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://example.invalid/compatible-mode/v1")
os.environ.setdefault("DASHSCOPE_MODEL", "qwen-test-model")
os.environ.setdefault("DASHSCOPE_EMBEDDING_MODEL", "qwen-test-embedding")
os.environ.setdefault("RAG_SCORE_THRESHOLD", "0.45")
# 显式置空：本地 .env 里有真实密码，不隔离的话测试会被鉴权挡住，
# 而且 CI 上（无 .env）和本地的行为会不一致。需要鉴权的测试自行 monkeypatch。
os.environ.setdefault("UI_PASSWORD", "")
# Gateway 的缓存在测试里必须关掉：多个用例会发同样的消息，命中缓存后
# 就不会调到 mock，断言「传了什么给模型」会拿到上一个用例的残留。
os.environ.setdefault("LLM_CACHE_TTL", "0")

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app as app_module
import llm
from rag import pipeline
from rag import store as rag_store


@pytest.fixture
def client(monkeypatch):
    """默认无向量库 → 纯大模型模式。需要 RAG 的测试请另外用 fake_retrieval。"""
    monkeypatch.setattr(pipeline, "_store", None)
    monkeypatch.setattr(pipeline, "_loaded", True)
    return TestClient(app_module.app)


@pytest.fixture
def mock_llm(monkeypatch):
    """替换掉真实的大模型调用，避免测试消耗额度、依赖网络。"""

    def _set(reply="mocked reply"):
        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 12
            completion_tokens = 34

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()   # 带上 usage，让 Gateway 的计费链路也被测到

        def _create(*args, **kwargs):
            _set.called_with = kwargs
            return _Resp()

        monkeypatch.setattr(
            llm.gateway.providers[0].client.chat.completions, "create", _create
        )
        return _set

    return _set


@pytest.fixture
def fake_retrieval(monkeypatch):
    """注入假的检索结果，让路由判定可以脱网测试。

    用法：fake_retrieval((0.9, "文章A"), (0.5, "文章B"))
    """

    def _set(*scored, raises: bool = False):
        # 假 store 需要带 _collection.count()，/health 会统计块数
        class _FakeCollection:
            def count(self):
                return len(scored)

        class _FakeStore:
            _collection = _FakeCollection()

        monkeypatch.setattr(pipeline, "_store", _FakeStore())
        monkeypatch.setattr(pipeline, "_loaded", True)

        if raises:
            def _boom(*a, **kw):
                raise RuntimeError("embedding 服务挂了")
            monkeypatch.setattr(rag_store, "search", _boom)
            return

        hits = [
            (
                Document(
                    page_content=f"正文 {title}",
                    metadata={"title": title, "url": f"https://x/{i}", "category": "分类"},
                ),
                score,
            )
            for i, (score, title) in enumerate(scored)
        ]
        monkeypatch.setattr(rag_store, "search", lambda store, q, k=4: hits)

    return _set


@pytest.fixture
def mock_llm_stream(monkeypatch):
    """替换流式调用，产出固定的思考 + 正文增量。"""

    def _set(thinking="想一想。", content="这是回答。"):
        def _stream(messages, temperature=0.7, caller=None):
            _stream.called_with = {"messages": messages, "temperature": temperature,
                                   "caller": caller}
            for ch in thinking:
                yield "thinking", ch
            for ch in content:
                yield "content", ch

        monkeypatch.setattr(llm, "chat_stream", _stream)
        return _stream

    return _set
