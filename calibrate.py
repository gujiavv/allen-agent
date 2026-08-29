# calibrate.py
"""标定 RAG_SCORE_THRESHOLD。执行：python calibrate.py

用两组问题实测分数分布，取能分开它们的值作为阈值。
不要凭感觉改阈值——改之前先跑这个脚本看数据。
"""
import sys

import config
from rag import store as rag_store

# 域内：能在 all-articles.md 里找到答案的问题，中英混合（用户多用中文，文档是英文）
IN_DOMAIN = [
    "How do I know if my Ballet product is genuine?",
    "我的 Ballet 卡片是正品吗，怎么验证？",
    "Does Ballet provide phone service?",
    "Ballet 有客服电话吗？",
    "How do I activate my Ballet wallet?",
    "怎么激活我的钱包？",
    "What is the shipping time for my order?",
    "订单多久能发货？",
    "How do I import a private key into the app?",
    "退货政策是什么？",
    "What is passphrase entropy?",
    "gas 费太高了怎么办？",
    "Can PURE Bitcoin be used for shopping?",
    "如何把比特币从实体币里转出来？",
    "有人自称 Ballet 客服要我的私钥，这是骗子吗？",
]

# 域外：知识库里根本没有的问题，应当退回纯大模型
OUT_OF_DOMAIN = [
    "今天天气怎么样？",
    "帮我写一首关于秋天的诗",
    "Python 里怎么读取 JSON 文件？",
    "1 加 1 等于几？",
    "推荐几部好看的科幻电影",
    "How do I cook pasta?",
    "北京到上海有多远？",
    "解释一下量子纠缠",
    "帮我写一封辞职信",
    "给我讲个笑话",
]


def main() -> int:
    store = rag_store.load()
    if store is None:
        print(f"❌ 未找到向量库 {config.VECTOR_STORE_DIR}，请先执行 python ingest.py")
        return 1

    def top1(q: str):
        hits = rag_store.search(store, q, k=config.RAG_TOP_K)
        return (hits[0][1], hits[0][0].metadata.get("title", "")) if hits else (0.0, "")

    print(f"{'域内问题（应走 RAG）':50s} 分数   命中文章")
    print("-" * 110)
    in_scores = []
    for q in IN_DOMAIN:
        s, t = top1(q)
        in_scores.append(s)
        print(f"{q:50s} {s:.4f}  {t[:45]}")

    print(f"\n{'域外问题（应退回大模型）':50s} 分数   命中文章")
    print("-" * 110)
    out_scores = []
    for q in OUT_OF_DOMAIN:
        s, t = top1(q)
        out_scores.append(s)
        print(f"{q:50s} {s:.4f}  {t[:45]}")

    lo_in, hi_out = min(in_scores), max(out_scores)
    print("\n" + "=" * 110)
    print(f"域内最低分: {lo_in:.4f}      域外最高分: {hi_out:.4f}")

    if lo_in > hi_out:
        suggested = round((lo_in + hi_out) / 2, 2)
        print(f"✅ 两组完全可分，间隔 {lo_in - hi_out:.4f}")
        print(f"   建议 RAG_SCORE_THRESHOLD = {suggested}")
    else:
        overlap_in = sum(1 for s in in_scores if s <= hi_out)
        overlap_out = sum(1 for s in out_scores if s >= lo_in)
        print(f"⚠️  两组有重叠：{overlap_in} 个域内问题分数不高于域外最高分，"
              f"{overlap_out} 个域外问题不低于域内最低分")
        # 重叠时偏向"宁可多走 RAG"，靠提示词兜底纠正，好过该查文档时没查
        suggested = round(hi_out + 0.01, 2)
        print(f"   建议 RAG_SCORE_THRESHOLD = {suggested}（压住全部域外，"
              f"漏掉的域内问题由提示词兜底）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
