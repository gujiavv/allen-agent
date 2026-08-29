# app.py
"""FastAPI 装配层：定义路由、挂载 Gradio 前端。业务逻辑都在 rag/ 和 llm.py 里。"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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


app = FastAPI(title="我的Agent服务", version="2.0", lifespan=lifespan)


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


@app.get("/health")
async def health():
    """健康检查接口，用于部署后验证服务是否正常"""
    return {"status": "ok"}


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

# 启动命令（本地测试用）：
# uvicorn app:app --host 0.0.0.0 --port 8000
# 页面地址：http://127.0.0.1:8000/ui
