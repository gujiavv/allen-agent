# eval/run_eval.py
"""在评测集上横向对比各检索方案。执行：python eval/run_eval.py [--rerank]

指标：
  Recall@1 / Recall@4  标准答案是否出现在前 1 / 前 4 条里
  MRR                  标准答案的排名倒数的均值，衡量"排得多靠前"
  拒答准确率           域外问题中，top1 分数低于阈值（即正确拒答）的比例
  误拒率               域内问题中，top1 分数低于阈值（本该答却拒答）的比例

拒答两项只对分数可比的方案有意义（BM25 和 RRF 的分数不是相关度，无法套用
0.45 这个阈值），所以那两栏对它们显示 n/a。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from eval import retrievers as R  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.json"
DUPLICATES = Path(__file__).resolve().parent / "duplicates.json"


def _load_duplicates() -> dict:
    """近重复文章映射。

    知识库本身有严重重复：194 篇里 63 篇存在相似度 ≥0.95 的孪生条目
    （抓取时同一篇被收了两遍，一份标题带 # 前缀）。标准答案是 A 而召回了
    它 99% 相同的孪生 A'，机械判定算未命中，但语义上完全正确。
    不做这个修正，测出来的是知识库的数据质量，不是检索能力。
    """
    if not DUPLICATES.exists():
        return {}
    return {k: set(v) for k, v in json.loads(
        DUPLICATES.read_text(encoding="utf-8")).items()}


def evaluate(retriever, rows, k=4, score_comparable=True, duplicates=None):
    """duplicates 非空时，命中标准答案的近重复条目也算命中。"""
    duplicates = duplicates or {}

    def _match(title: str, gold: str) -> bool:
        return title == gold or title in duplicates.get(gold, ())

    in_rows = [r for r in rows if r["in_domain"]]
    out_rows = [r for r in rows if not r["in_domain"]]

    hit1 = hit_k = 0
    rr_sum = 0.0
    latencies = []
    false_refuse = 0

    for row in in_rows:
        t0 = time.perf_counter()
        results = retriever(row["question"], k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        titles = [t.strip() for t, _ in results]
        gold = row["expected_title"].strip()
        if titles and _match(titles[0], gold):
            hit1 += 1
        rank = next((i for i, t in enumerate(titles) if _match(t, gold)), None)
        if rank is not None:
            hit_k += 1
            rr_sum += 1.0 / (rank + 1)
        if score_comparable and results and results[0][1] < config.RAG_SCORE_THRESHOLD:
            false_refuse += 1

    correct_refuse = 0
    if score_comparable:
        for row in out_rows:
            results = retriever(row["question"], k=k)
            if not results or results[0][1] < config.RAG_SCORE_THRESHOLD:
                correct_refuse += 1

    n = len(in_rows)
    return {
        "方案": retriever.name,
        "Recall@1": hit1 / n,
        f"Recall@{k}": hit_k / n,
        "MRR": rr_sum / n,
        "拒答准确率": correct_refuse / len(out_rows) if score_comparable else None,
        "误拒率": false_refuse / n if score_comparable else None,
        "延迟中位(ms)": statistics.median(latencies),
    }


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.1%}" if isinstance(v, float) and v <= 1 else f"{v:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true",
                    help="加入本地 rerank 对比（需要 sentence-transformers + torch）")
    args = ap.parse_args()

    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    print(f"评测集：{len(rows)} 条"
          f"（域内 {sum(1 for r in rows if r['in_domain'])} / "
          f"域外 {sum(1 for r in rows if not r['in_domain'])}）")
    print(f"阈值 RAG_SCORE_THRESHOLD = {config.RAG_SCORE_THRESHOLD}\n")

    vector = R.VectorRetriever()
    bm25 = R.BM25Retriever()

    plans = [(vector, True), (bm25, False)]
    for w in (0.3, 0.5, 0.7, 0.9):
        plans.append((R.HybridRetriever(vector, bm25, w_vector=w), False))

    if args.rerank:
        plans.append((R.RerankRetriever(vector), False))

    dups = _load_duplicates()
    if dups:
        print(f"已加载近重复映射：{len(dups)} 篇文章有孪生条目，"
              f"命中孪生条目同样计为命中\n")

    results = []
    for retriever, comparable in plans:
        print(f"  跑 {retriever.name} ...", flush=True)
        results.append(evaluate(retriever, rows, score_comparable=comparable,
                                duplicates=dups))

    cols = ["方案", "Recall@1", "Recall@4", "MRR", "拒答准确率", "误拒率", "延迟中位(ms)"]
    widths = [max(len(c), max(len(_fmt(r[c])) if c != "方案" else len(r[c])
                              for r in results)) + 2 for c in cols]
    print("\n" + "".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * sum(widths))
    for r in results:
        print("".join((r[c] if c == "方案" else _fmt(r[c])).ljust(w)
                      for c, w in zip(cols, widths)))

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
