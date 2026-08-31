# ui.py
"""Gradio 聊天页面，挂载在 /ui。

页面不直接 import pipeline，而是发一个真实的 HTTP 请求打到本服务的 /chat，
这样页面走的代码路径（路由、pydantic 校验、响应序列化）和外部调用者完全一致，
不会出现"页面能用但接口是坏的"这种偏差。

请求通过 httpx 的 ASGITransport 直接投递给 app 对象，不经过 TCP。
早先的写法是 POST 到 http://127.0.0.1:{PORT}/chat，但那要靠 PORT 环境变量去猜
端口，而 uvicorn 的真实端口来自 --port 参数——两者不一致时（例如本地
`uvicorn --port 8123` 却没设 PORT），自调用会静默打到 8000 上**别的**服务，
甚至把用户提问发给毫不相干的第三方。ASGITransport 从根上消除了这个风险。
"""
import json
import logging
import os
import secrets
import time

# 必须在 import gradio 之前设置：Gradio 默认会在导入时向 api.gradio.app
# 上报遥测并检查版本，服务端不需要这种外连，CI 里也会拖慢并污染日志。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import httpx  # noqa: E402

import config  # noqa: E402

logger = logging.getLogger(__name__)

# ASGITransport 不做 DNS 解析，host 只是个占位符
CHAT_URL = "http://asgi.internal/chat"
STREAM_URL = "http://asgi.internal/chat/stream"

_client: httpx.AsyncClient | None = None

DESCRIPTION = (
    "问 Ballet 冷钱包相关问题会自动检索官方支持文档并附上原文链接；"
    "文档里没有的问题则由大模型直接回答。"
)

EXAMPLES = [
    "我的 Ballet 卡片是正品吗，怎么验证？",
    "Ballet 有客服电话吗？",
    "订单多久能发货？",
    "有人自称 Ballet 客服要我的私钥，这是骗子吗？",
]


async def _respond(message: str, history):
    """异步生成器：每收到一点增量就 yield 一次，Gradio 会实时刷新。

    产出的是 gr.ChatMessage 列表——思考过程带 metadata，Gradio 会把它渲染成
    可折叠的面板，默认收起，点开才看得到，不至于把正式回答挤下去。
    """
    if not message or not message.strip():
        yield "请输入问题。"
        return

    thinking, content, sources, mode = "", "", [], "llm"
    started = time.monotonic()

    def _render():
        blocks = []
        if thinking:
            done = bool(content)
            blocks.append(
                gr.ChatMessage(
                    role="assistant",
                    content=thinking,
                    metadata={
                        "title": "💭 思考完成" if done else "💭 正在思考…",
                        "status": "done" if done else "pending",
                        "duration": round(time.monotonic() - started, 1),
                    },
                )
            )
        if content:
            blocks.append(gr.ChatMessage(role="assistant", content=content + _footer()))
        return blocks

    def _footer() -> str:
        if mode == "rag" and sources:
            lines = ["", "", "---", "📚 **依据以下 Ballet 官方文档回答：**"]
            for src in sources:
                title, url, score = src.get("title", ""), src.get("url", ""), src.get("score", 0)
                lines.append(
                    f"- [{title}]({url}) · 相关度 {score}" if url
                    else f"- {title} · 相关度 {score}"
                )
            return "\n".join(lines)
        return "\n\n---\n🤖 *知识库中没有相关文档，以上由大模型自身知识作答。*"

    try:
        # 页面自调用也要过鉴权：/chat/stream 对所有调用方一视同仁，
        # 不给内部调用开后门，省得哪天后门比正门还好走。
        headers = (
            {"X-API-Password": config.UI_PASSWORD} if config.UI_PASSWORD else {}
        )
        async with _client.stream(
            "POST", STREAM_URL, json={"message": message}, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                kind, value = event["type"], event["value"]

                if kind == "mode":
                    mode = value
                elif kind == "sources":
                    sources = value
                elif kind == "thinking":
                    thinking += value
                    yield _render()
                elif kind == "content":
                    content += value
                    yield _render()
                elif kind == "error":
                    yield f"⚠️ 出错了：{value}"
                    return
                elif kind == "done":
                    break
    except Exception as e:
        yield f"⚠️ 请求失败：{e}"
        return

    # 收尾：思考面板置为完成态，正文补上引用
    if not content and not thinking:
        yield "（模型没有返回内容）"
    else:
        yield _render()


def build_demo() -> gr.Blocks:
    return gr.ChatInterface(
        fn=_respond,
        title="🩰 Ballet 智能客服",
        description=DESCRIPTION,
        examples=EXAMPLES,
        analytics_enabled=False,
    )


def _check_login(username: str, password: str) -> bool:
    """只校验密码，用户名随便填。

    compare_digest 是定时安全比较，避免响应耗时泄漏已猜对多少位。
    """
    return bool(config.UI_PASSWORD) and secrets.compare_digest(
        password, config.UI_PASSWORD
    )


def mount_ui(app):
    """把 Gradio 页面挂到传入的 FastAPI 实例的 /ui 路径上。

    配了 UI_PASSWORD 就要求登录；没配则直接开放，方便本地开发。
    线上是否真的启用了，查 /health 的 auth_enabled 字段。
    """
    global _client
    _client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://asgi.internal",
        timeout=90.0,
    )
    kwargs = {}
    if config.UI_PASSWORD:
        kwargs["auth"] = _check_login
        kwargs["auth_message"] = "用户名可留空或随便填，输入访问密码即可进入。"
    else:
        logger.warning(
            "未设置 UI_PASSWORD，/ui 与 /chat 完全开放。"
            "公开部署请务必在环境变量中配置密码。"
        )
    return gr.mount_gradio_app(app, build_demo(), path="/ui", **kwargs)
