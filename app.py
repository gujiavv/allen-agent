# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx
from openai import OpenAI

# 环境变量管理（用你的真实API Key）
# export DEEPSEEK_API_KEY="sk-xxx"  # 命令行设置
# 或者用 .env 文件

app = FastAPI(title="我的Agent服务", version="1.0")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")

if not api_key:
    raise ValueError("请在项目根目录的 .env 文件中配置 DEEPSEEK_API_KEY")

# 初始化大模型客户端（以 DeepSeek 为例）
# trust_env=False: 本机全局代理(HTTPS_PROXY)到 api.deepseek.com 的 TLS 握手会失败，
# 所以让 HTTP 客户端直连、忽略环境变量里的代理。若你的网络必须走代理，改成 True。
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=httpx.Client(trust_env=False, timeout=60.0),
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """核心Agent接口：接收消息，调用大模型，返回回复"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.message}],
            temperature=0.7
        )
        reply = response.choices[0].message.content
        return ChatResponse(reply=reply)
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

# 启动命令（本地测试用）：
# uvicorn app:app --host 0.0.0.0 --port 8000
