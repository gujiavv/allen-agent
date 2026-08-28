"""测试夹具。

app.py 在 import 阶段就会校验 DEEPSEEK_API_KEY，
所以必须在导入 app 之前把环境变量准备好。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-key-for-ci")
os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def mock_llm(monkeypatch):
    """替换掉真实的大模型调用，避免测试消耗额度、依赖网络。"""

    def _set(reply="mocked reply"):
        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        def _create(*args, **kwargs):
            _set.called_with = kwargs
            return _Resp()

        monkeypatch.setattr(app_module.client.chat.completions, "create", _create)
        return _set

    return _set
