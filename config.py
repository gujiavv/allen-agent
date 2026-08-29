# config.py
"""环境变量与常量的集中读取、校验。

原先这些散在 app.py 顶部，但加入 RAG 后 ingest.py / calibrate.py / ui.py
也要读同一批配置，所以抽出来单点管理，避免各处 getenv 拼写不一致。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---- 百炼（DashScope）----
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
CHAT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding")

# 百炼 embedding 接口一次最多 10 条，第 11 条起返回
# 400 InternalError.Algo.InvalidParameter: batch size is invalid
EMBEDDING_BATCH_SIZE = 10

# ---- 知识库 ----
KNOWLEDGE_FILE = BASE_DIR / "all-articles.md"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
COLLECTION_NAME = "ballet_support"

# ---- 检索参数 ----
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# 相关性分数低于此值即判定"文档里查不到"，退回纯大模型作答。
# 该默认值由 calibrate.py 用域内/域外两组问题实测标定，勿凭感觉改。
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))

if not API_KEY:
    raise ValueError(
        "缺少 DASHSCOPE_API_KEY。本地开发请在项目根目录 .env 中配置；"
        "部署到 Railway 等平台请在平台的环境变量面板中配置。"
    )
