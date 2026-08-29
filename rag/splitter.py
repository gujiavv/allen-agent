# rag/splitter.py
"""把 all-articles.md 切成带元数据的块。

纯函数、不联网，因此可以脱网单元测试。
文档天然是三级结构：# 分类 / ## 小节 / ### 文章，直接拿来做元数据。
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADERS = [("#", "category"), ("##", "section"), ("###", "title")]

# 只有超过这个长度的文章才二次切分。
# 必须按字面实现，不可简化成"对所有块统一跑 chunk_size=800"——文章长度中位数
# 是 1053，正好落在 800 和 1500 之间，那样会把大量普通文章无谓劈成两半。
LONG_ARTICLE_CHARS = 1500
SUB_CHUNK_SIZE = 800
SUB_CHUNK_OVERLAP = 120

# 每篇文章标题下方那行原文链接，形如 <https://support.ballet.com/hc/en-us/articles/...>
URL_RE = re.compile(r"<(https?://[^>\s]+)>")


def _extract_url(text: str) -> str:
    """取正文里第一个尖括号链接作为该文章的原文地址。"""
    m = URL_RE.search(text)
    return m.group(1) if m else ""


def _context_prefix(meta: dict) -> str:
    """`分类 > 小节 > 标题`，用于给子块补回主题。"""
    parts = [meta.get(k) for k in ("category", "section", "title")]
    return " > ".join(p for p in parts if p)


def split_documents(markdown_text: str) -> list[Document]:
    """按三级标题切分，长文再切，产出可直接入库的 Document 列表。"""
    md_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUB_CHUNK_SIZE, chunk_overlap=SUB_CHUNK_OVERLAP
    )

    documents: list[Document] = []
    for article in md_splitter.split_text(markdown_text):
        meta = dict(article.metadata)

        # 没有 title 的是分类/小节下的引言片段，不是文章，跳过
        if not meta.get("title"):
            continue

        url = _extract_url(article.page_content)
        # 链接已存进元数据，从正文里去掉，省得占 embedding 的语义空间
        body = URL_RE.sub("", article.page_content).strip()

        meta["url"] = url
        meta.setdefault("category", "")
        meta.setdefault("section", "")
        prefix = _context_prefix(meta)

        # 少数文章在原文里只有链接、没有正文（抓取时正文缺失）。这类仍然入库，
        # 只嵌入"分类 > 小节 > 标题"，这样用户问到时至少能把原文链接推出来。
        if not body:
            pieces = [""]
        elif len(body) > LONG_ARTICLE_CHARS:
            pieces = sub_splitter.split_text(body)
        else:
            pieces = [body]

        for i, piece in enumerate(pieces):
            content = f"{prefix}\n\n{piece}".strip() if piece.strip() else prefix
            # 内容为空的块必须丢掉：百炼 embedding 对空字符串返回 2560 维
            # （而非 1024），混进去会让 Chroma 报维度冲突。
            if not content.strip():
                continue
            documents.append(
                Document(
                    # 前置"分类 > 小节 > 标题"，避免长文中段的子块脱离主题。
                    # 用户看到的引用来自 metadata 的 title/url，不是这段文本。
                    page_content=content,
                    metadata={**meta, "chunk_index": i, "chunk_total": len(pieces)},
                )
            )

    return documents


def load_and_split(path) -> list[Document]:
    """从文件读取并切分。"""
    return split_documents(open(path, encoding="utf-8").read())
