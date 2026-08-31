# app.py
"""FastAPI 装配层：定义路由、挂载 Gradio 前端。业务逻辑都在 rag/ 和 llm.py 里。"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from rag import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载向量库。加载不到不影响服务启动，只是降级为纯大模型模式。"""
    if pipeline.init():
        logger.info("向量库已加载，RAG 生效（阈值 %.2f）", config.RAG_SCORE_THRESHOLD)
    yield


app = FastAPI(title="我的Agent服务", version=config.APP_VERSION, lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    title: str
    url: str
    category: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    # 走了哪条路：rag=基于知识库作答，llm=知识库里没有，用模型自身知识
    mode: str = "llm"
    sources: list[Source] = []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """核心Agent接口：能从知识库检索到就用文档回答，否则退回大模型自身知识。"""
    try:
        reply, mode, sources = pipeline.answer(request.message)
        return ChatResponse(reply=reply, mode=mode, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """/chat 的流式版本，用 SSE 逐段推送。

    单独开一个端点而不是给 /chat 加 stream 参数：/chat 的响应契约已经被测试和
    外部调用方锁住了，同一个端点返回两种形态会让契约变得含混。

    每行形如 `data: {"type": ..., "value": ...}`，type 取值：
    mode / sources / thinking / content / error / done
    """

    def _sse(event_type: str, value) -> str:
        return "data: " + json.dumps(
            {"type": event_type, "value": value}, ensure_ascii=False
        ) + "\n\n"

    def _generate():
        try:
            for event_type, value in pipeline.answer_stream(request.message):
                yield _sse(event_type, value)
        except Exception as e:
            logger.exception("流式回答失败")
            yield _sse("error", str(e))
        finally:
            yield _sse("done", None)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        # 关掉 Nginx 一类反向代理的缓冲，否则流式会被攒成一坨再发出来
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """健康检查，同时自报版本与 RAG 状态。

    加上 commit 是为了能一眼确认线上跑的到底是哪次提交，不用去翻部署面板。
    加上 rag_enabled 是因为索引没进镜像时服务照样能起、health 照样返回 ok，
    RAG 却是死的——这个字段让那种静默降级藏不住。
    """
    # 健康检查绝不能崩：Railway 靠它判断服务是否存活，这里抛异常会让整个部署
    # 被标记为失败。统计块数用到了 Chroma 的私有属性，所以整段包在 try 里。
    rag_enabled, chunks = False, 0
    try:
        store = pipeline.get_store()
        if store is not None:
            rag_enabled = True
            chunks = store._collection.count()
    except Exception:
        logger.exception("统计向量库状态失败，不影响健康检查结果")

    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "commit": config.COMMIT,
        "branch": config.BRANCH or None,
        "rag_enabled": rag_enabled,
        "chunks": chunks,
    }


@app.get("/myapp")
async def myapp():
    """健康检查接口，用于部署后验证服务是否正常"""
    return {
            "status": "ok",
            "body": {
                "name":"张三丰",
                },
            }


# Gradio 页面挂在 /ui，与本服务同进程同端口。
# 放在文件末尾：此时 app 上的路由都已注册完毕。
from ui import mount_ui  # noqa: E402

mount_ui(app)

if __name__ == "__main__":
    # 支持直接 `python app.py` 启动。注意必须用项目 venv 里的解释器：
    #     .venv/bin/python app.py
    # 系统 Python 没装 gradio / langchain，会报 ModuleNotFoundError。
    #
    # 端口在这里就地读取，不放进 config.py：Gradio 页面改用 ASGITransport 之后
    # 全项目只有这一处需要知道端口，放进配置模块反而会让人以为别处也依赖它。
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
