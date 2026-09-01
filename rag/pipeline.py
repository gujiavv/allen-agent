# rag/pipeline.py
"""编排：检索 → 阈值判定 → 组装提示词 → 产出回答与引用。

路由策略（自动，不暴露开关给调用方）：
  top1 相关性分数 ≥ 阈值 → 带文档回答，mode="rag"
  否则                    → 直接问大模型，mode="llm"
提示词里另有一道兜底：即使分数过阈但资料实际答不了，也允许模型改用自身知识。
"""
import logging

import config
import llm
from rag import store as rag_store

logger = logging.getLogger(__name__)

SYSTEM_RAG = """你是 Ballet 冷钱包的官方支持助手。

下面是从 Ballet 官方支持文档中检索到的资料。请遵守：

1. 若资料能回答用户的问题，就严格基于资料作答，不要编造资料中没有的细节。
2. 若资料与问题无关、或不足以回答，请直接说明"Ballet 支持文档中未涵盖这个问题"，
   然后用你自己的知识作答。**不要硬从无关资料里凑答案。**
3. 用户用什么语言提问，就用什么语言回答（资料是英文的，中文提问请用中文回答）。
4. 涉及资产安全的问题要谨慎：Ballet 永远不会索要私钥熵或口令熵。

检索到的资料：
{context}"""

_store = None
_loaded = False


def init() -> bool:
    """启动时调用一次。返回向量库是否可用。"""
    global _store, _loaded
    _loaded = True
    _store = rag_store.load()
    if _store is None:
        logger.warning(
            "未找到向量库 %s，将以纯大模型模式运行（RAG 不生效）。"
            "请先执行 `python ingest.py` 建立索引。",
            config.VECTOR_STORE_DIR,
        )
        return False
    return True


def _get_store():
    if not _loaded:
        init()
    return _store


def get_store():
    """供 /health 查询向量库状态。"""
    return _get_store()


def _format_context(hits) -> str:
    blocks = []
    for i, (doc, score) in enumerate(hits, 1):
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        blocks.append(f"【资料 {i}】{title}\n来源：{url}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _to_sources(hits) -> list[dict]:
    seen, sources = set(), []
    for doc, score in hits:
        title = doc.metadata.get("title", "")
        if title in seen:  # 长文的多个子块可能同时命中，按文章去重
            continue
        seen.add(title)
        sources.append(
            {
                "title": title,
                "url": doc.metadata.get("url", ""),
                "category": doc.metadata.get("category", ""),
                "score": round(float(score), 4),
            }
        )
    return sources


def _plain_answer(message: str, caller=None) -> tuple[str, str, list]:
    return llm.chat([{"role": "user", "content": message}], caller=caller), "llm", []


def answer(message: str, caller: str | None = None) -> tuple[str, str, list[dict]]:
    """返回 (回复文本, 模式, 引用列表)。模式为 "rag" 或 "llm"。"""
    store = _get_store()
    if store is None:
        return _plain_answer(message, caller)

    try:
        hits = rag_store.search(store, message, k=config.RAG_TOP_K)
    except Exception:
        # 检索链路（embedding 接口）出问题不该让整个请求失败，降级继续服务
        logger.exception("检索失败，降级为纯大模型回答")
        return _plain_answer(message, caller)

    if not hits or hits[0][1] < config.RAG_SCORE_THRESHOLD:
        return _plain_answer(message, caller)

    # top1 过阈才走这条路；上下文只收同样过阈的块，避免塞进不相关内容
    relevant = [h for h in hits if h[1] >= config.RAG_SCORE_THRESHOLD]
    reply = llm.chat(
        [
            {"role": "system", "content": SYSTEM_RAG.format(context=_format_context(relevant))},
            {"role": "user", "content": message},
        ],
        temperature=0.3,  # 有资料时要贴着资料答，别发挥
        caller=caller,
    )
    return reply, "rag", _to_sources(relevant)


def answer_stream(message: str, caller: str | None = None):
    """流式版本，逐个产出事件元组：

        ("sources", [...])   检索完成，先把命中的文档推给前端
        ("mode", "rag"/"llm") 走了哪条路
        ("thinking", str)    模型推理过程的增量
        ("content", str)     正式回答的增量

    路由判定逻辑与 answer() 完全一致，只是把结果改成边生成边吐。
    """
    store = _get_store()
    hits = None

    if store is not None:
        try:
            hits = rag_store.search(store, message, k=config.RAG_TOP_K)
        except Exception:
            logger.exception("检索失败，降级为纯大模型回答")
            hits = None

    use_rag = bool(hits) and hits[0][1] >= config.RAG_SCORE_THRESHOLD

    if use_rag:
        relevant = [h for h in hits if h[1] >= config.RAG_SCORE_THRESHOLD]
        yield "mode", "rag"
        yield "sources", _to_sources(relevant)
        messages = [
            {"role": "system", "content": SYSTEM_RAG.format(context=_format_context(relevant))},
            {"role": "user", "content": message},
        ]
        temperature = 0.3
    else:
        yield "mode", "llm"
        yield "sources", []
        messages = [{"role": "user", "content": message}]
        temperature = 0.7

    yield from llm.chat_stream(messages, temperature=temperature, caller=caller)
