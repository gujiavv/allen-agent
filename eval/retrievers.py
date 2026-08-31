# eval/retrievers.py
"""几种检索策略，接口统一，便于在同一批问题上横向对比。

每个检索器都是 callable：query -> [(title, score)]，按相关性降序。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag import splitter, store as rag_store  # noqa: E402


def dedupe_by_article(pairs):
    """把块级结果收敛成文章级：同一篇文章只保留排名最靠前的那个块。

    标准答案是文章级的，所有方案都必须在同一粒度上比较才公平。
    更要紧的是：不去重就会让块多的长文章在融合时累加出虚高的分数
    （一篇 3 块的文章能靠累加压过只有 1 块但排第一的文章）。
    """
    seen, out = set(), []
    for title, score in pairs:
        t = title.strip()
        if t in seen:
            continue
        seen.add(t)
        out.append((t, score))
    return out

_CJK = r"一-鿿"


def tokenize(text: str) -> list[str]:
    """给 BM25 用的分词。

    英文按词切；中文按「单字 + 相邻二元组」切——二元组能让"客服""电话"这类
    词语得到匹配机会。刻意做得公平一些：如果只按单字切，BM25 会因为中文单字
    区分度太低而输得很难看，那样的对比是稻草人。
    """
    text = text.lower()
    tokens = re.findall(rf"[a-z0-9]+|[{_CJK}]", text)
    cjk = [t for t in tokens if re.match(rf"[{_CJK}]", t)]
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens + bigrams


class VectorRetriever:
    """当前线上用的：Chroma 向量检索。"""

    name = "纯向量"

    def __init__(self):
        self.store = rag_store.load()
        if self.store is None:
            raise RuntimeError("向量库不存在，请先执行 python ingest.py")

    def __call__(self, query: str, k: int = 4):
        # 多取一些再去重，否则去重后可能凑不满 k 篇
        hits = self.store.similarity_search_with_relevance_scores(query, k=k * 4)
        return dedupe_by_article(
            [(d.metadata["title"], s) for d, s in hits])[:k]


class BM25Retriever:
    """词项精确匹配。语料是英文、提问是中文，理论上会很吃亏——这正是要验证的。"""

    name = "BM25"

    def __init__(self, docs=None):
        from rank_bm25 import BM25Okapi

        self.docs = docs if docs is not None else splitter.load_and_split(
            config.KNOWLEDGE_FILE)
        self.titles = [d.metadata["title"].strip() for d in self.docs]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    def __call__(self, query: str, k: int = 4):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 4]
        return dedupe_by_article(
            [(self.titles[i], float(scores[i])) for i in order])[:k]


class HybridRetriever:
    """向量 + BM25 融合，用 RRF（倒数排名融合）。

    选 RRF 而不是加权分数相加：两者的分数不同量纲（向量是 0~1 的相关度，
    BM25 是无上界的词频得分），直接加权相加没有意义，必须先归一化，
    而归一化方式本身又会引入新的超参。RRF 只用排名，天然免疫量纲问题。
    """

    def __init__(self, vector, bm25, w_vector=0.5, rrf_k=60):
        self.vector, self.bm25 = vector, bm25
        self.w_vector, self.rrf_k = w_vector, rrf_k
        self.name = f"混合(向量{w_vector:.1f}/BM25{1-w_vector:.1f})"

    def __call__(self, query: str, k: int = 4):
        pool = 20  # 各自多取一些再融合，否则融合没有发挥空间
        fused: dict[str, float] = {}
        for retriever, weight in (
            (self.vector, self.w_vector), (self.bm25, 1 - self.w_vector)
        ):
            # retriever 返回的已是文章级去重结果，所以每篇文章在这里只贡献一次，
            # 不会出现块多的长文靠累加虚高的问题。
            for rank, (title, _) in enumerate(retriever(query, k=pool)):
                fused[title] = fused.get(title, 0.0) + weight / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]


class RerankRetriever:
    """先用向量粗召回，再用 cross-encoder 精排。

    百炼该端点没有 rerank 模型（247 个模型里一个都没有），所以只能用本地模型。
    代价是引入 torch，镜像会涨到 3GB 左右——因此本地实验专用，不进部署镜像。
    """

    def __init__(self, base, model_name="BAAI/bge-reranker-v2-m3", pool=20):
        from sentence_transformers import CrossEncoder

        self.base, self.pool = base, pool
        self.model = CrossEncoder(model_name)
        self.name = f"向量+Rerank(取{pool}精排)"
        self.docs = {}
        for d in splitter.load_and_split(config.KNOWLEDGE_FILE):
            self.docs.setdefault(d.metadata["title"].strip(), d.page_content)

    def __call__(self, query: str, k: int = 4):
        uniq = [t for t, _ in self.base(query, k=self.pool)]
        if not uniq:
            return []
        scores = self.model.predict([(query, self.docs.get(t, t)) for t in uniq])
        ranked = sorted(zip(uniq, scores), key=lambda kv: kv[1], reverse=True)
        return [(t, float(s)) for t, s in ranked[:k]]
