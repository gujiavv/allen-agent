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
import os

# 必须在 import gradio 之前设置：Gradio 默认会在导入时向 api.gradio.app
# 上报遥测并检查版本，服务端不需要这种外连，CI 里也会拖慢并污染日志。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import httpx  # noqa: E402

# ASGITransport 不做 DNS 解析，host 只是个占位符
CHAT_URL = "http://asgi.internal/chat"

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


async def _respond(message: str, history) -> str:
    if not message or not message.strip():
        return "请输入问题。"

    try:
        response = await _client.post(CHAT_URL, json={"message": message})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"⚠️ 请求失败：{e}"

    reply = data.get("reply", "")
    sources = data.get("sources", [])

    if data.get("mode") == "rag" and sources:
        lines = ["", "", "---", "📚 **依据以下 Ballet 官方文档回答：**"]
        for s in sources:
            title, url, score = s.get("title", ""), s.get("url", ""), s.get("score", 0)
            # 少数文章原文里没有链接，此时只显示标题，不渲染成坏链接
            lines.append(
                f"- [{title}]({url}) · 相关度 {score}" if url
                else f"- {title} · 相关度 {score}"
            )
        reply += "\n".join(lines)
    else:
        reply += "\n\n---\n🤖 *知识库中没有相关文档，以上由大模型自身知识作答。*"

    return reply


def build_demo() -> gr.Blocks:
    return gr.ChatInterface(
        fn=_respond,
        title="🩰 Ballet 智能客服",
        description=DESCRIPTION,
        examples=EXAMPLES,
        analytics_enabled=False,
    )


def mount_ui(app):
    """把 Gradio 页面挂到传入的 FastAPI 实例的 /ui 路径上。"""
    global _client
    _client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://asgi.internal",
        timeout=90.0,
    )
    return gr.mount_gradio_app(app, build_demo(), path="/ui")
