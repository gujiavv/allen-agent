# rag/store.py
"""Chroma 向量库的建立、加载与检索。只管存取，不含业务判定。"""
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        # 百炼一次最多收 10 条，超过直接 400
        chunk_size=config.EMBEDDING_BATCH_SIZE,
        # 必须关闭：开启时 LangChain 会把文本转成 token 数组再发，
        # 而百炼只接受字符串数组，会返回
        # 400 "input must be an array of strings"
        check_embedding_ctx_length=False,
    )


def _common_kwargs() -> dict:
    return dict(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.VECTOR_STORE_DIR),
        # 显式用余弦距离，这样 relevance score 落在 0~1 且方向直观（越大越相关）
        collection_metadata={"hnsw:space": "cosine"},
    )


def build(documents: list[Document], batch_size: int = config.EMBEDDING_BATCH_SIZE,
          on_progress=None) -> Chroma:
    """重建向量库。会先删掉旧目录，保证不残留上一次的块。"""
    if config.VECTOR_STORE_DIR.exists():
        shutil.rmtree(config.VECTOR_STORE_DIR)

    store = Chroma(**_common_kwargs())
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(batch)
        if on_progress:
            on_progress(min(start + batch_size, len(documents)), len(documents))
    return store


def load() -> Chroma | None:
    """加载已落盘的向量库；不存在或为空则返回 None，由调用方降级处理。"""
    if not config.VECTOR_STORE_DIR.exists():
        return None
    try:
        store = Chroma(**_common_kwargs())
        if store._collection.count() == 0:
            return None
        return store
    except Exception:
        return None


def search(store: Chroma, query: str, k: int = config.RAG_TOP_K):
    """返回 [(Document, 相关性分数)]，分数越大越相关。"""
    return store.similarity_search_with_relevance_scores(query, k=k)
