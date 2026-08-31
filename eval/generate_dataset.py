# eval/generate_dataset.py
"""从知识库反向生成检索评测集。执行：python eval/generate_dataset.py

做法：把每篇文章交给大模型，让它写出用户可能会问的问题。该文章就是这些问题的
标准答案（ground truth）。这是业界常用的合成评测集做法。

【必须知道的局限】
合成问题源自文章本身，会不自觉带上文章的词汇和表述，比真实用户的提问"干净"，
因此**评测结果会系统性高估真实场景的检索性能**。它能用来做「方案 A vs 方案 B」
的横向对比（各方案面对同一批问题，偏差一致），但不能用来宣称绝对准确率。
真正可靠的评测集要从真实用户提问里抽样、由不写代码的人标注。
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import llm  # noqa: E402
from rag import splitter  # noqa: E402

OUT = Path(__file__).resolve().parent / "dataset.json"
QUESTIONS_PER_ARTICLE = 2
WORKERS = 8

PROMPT = """下面是一篇 Ballet 冷钱包支持文档。请站在**真实用户**的角度，写出 {n} 个他们可能会问的问题。

要求：
1. **用中文提问**（真实用户以中文为主，而文档是英文的，这正是要考察的跨语言检索）。
2. **口语化**，像用户在客服窗口里打字，不要照抄文档里的措辞和术语。
   反例（照抄）："如何验证 Ballet 产品的真伪性？"
   正例（口语）："我买的这个卡是真的吗？怎么看出来"
3. 每个问题必须**只靠这篇文档就能回答**，不要问文档里没有的内容。
4. 问题之间角度要不同，不要换个说法问同一件事。

只输出 JSON 数组，不要任何解释：["问题1", "问题2"]

文档标题：{title}
文档内容：
{body}"""


def _gen_for_article(title: str, url: str, category: str, body: str):
    try:
        raw = llm.chat(
            [{"role": "user", "content": PROMPT.format(
                n=QUESTIONS_PER_ARTICLE, title=title, body=body[:3000])}],
            temperature=0.8,  # 要多样性，别每篇都问出同一个句式
        )
        m = re.search(r"\[.*\]", raw, re.S)
        questions = json.loads(m.group(0)) if m else []
    except Exception as e:
        print(f"\n  ⚠️ 《{title[:30]}》生成失败: {type(e).__name__}")
        return []

    return [
        {"question": q.strip(), "expected_title": title,
         "expected_url": url, "category": category, "in_domain": True}
        for q in questions if isinstance(q, str) and q.strip()
    ]


# 域外问题：知识库里没有，用来测「该拒答时有没有拒答」。
# 这些是手写的，不是生成的——域外问题不需要贴合任何文档。
OUT_OF_DOMAIN = [
    "今天北京天气怎么样？", "帮我写一首关于秋天的诗", "Python 怎么读取 JSON 文件？",
    "1 加 1 等于几？", "推荐几部好看的科幻电影", "How do I cook pasta?",
    "北京到上海有多远？", "解释一下量子纠缠", "帮我写一封辞职信", "给我讲个笑话",
    "特斯拉股票现在多少钱？", "怎么减肥最快？", "明天是星期几？",
    "帮我翻译一句英文", "世界上最高的山是哪座？",
]


def main() -> int:
    docs = splitter.load_and_split(config.KNOWLEDGE_FILE)

    # 按文章聚合：同一篇文章的多个块拼回去，标准答案是文章级别的
    articles: dict[str, dict] = {}
    for d in docs:
        t = d.metadata["title"].strip()
        if t not in articles:
            articles[t] = {"url": d.metadata.get("url", ""),
                           "category": d.metadata.get("category", ""), "body": ""}
        articles[t]["body"] += d.page_content + "\n"

    print(f"共 {len(articles)} 篇文章，每篇生成 {QUESTIONS_PER_ARTICLE} 个问题"
          f"（{WORKERS} 并发）...")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_gen_for_article, t, a["url"], a["category"], a["body"]): t
            for t, a in articles.items()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            print(f"\r  {i}/{len(futures)} 篇，已生成 {len(rows)} 条", end="", flush=True)

    rows.extend({"question": q, "expected_title": None, "expected_url": None,
                 "category": None, "in_domain": False} for q in OUT_OF_DOMAIN)

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    in_n = sum(1 for r in rows if r["in_domain"])
    print(f"\n✅ 已写入 {OUT}")
    print(f"   域内 {in_n} 条（覆盖 {len({r['expected_title'] for r in rows if r['in_domain']})} 篇文章）"
          f" | 域外 {len(rows) - in_n} 条 | 合计 {len(rows)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
