"""切块逻辑测试。纯函数、不联网。"""
from rag import splitter

SAMPLE = """# 分类甲

## 小节一

### 普通文章

<https://support.ballet.com/hc/en-us/articles/111-normal>

这是一篇短文章的正文。

### 只有链接没有正文的文章

<https://support.ballet.com/hc/en-us/articles/222-linkonly>

### 没有链接的文章

这篇文章在原文里就没有来源链接。
"""


def _by_title(docs, title):
    return [d for d in docs if d.metadata["title"].strip() == title]


def test_三级标题都进了元数据():
    docs = splitter.split_documents(SAMPLE)
    d = _by_title(docs, "普通文章")[0]
    assert d.metadata["category"] == "分类甲"
    assert d.metadata["section"] == "小节一"
    assert d.metadata["title"] == "普通文章"


def test_原文链接被抽进元数据且从正文移除():
    d = _by_title(splitter.split_documents(SAMPLE), "普通文章")[0]
    assert d.metadata["url"] == "https://support.ballet.com/hc/en-us/articles/111-normal"
    assert "support.ballet.com" not in d.page_content


def test_正文前置了分类小节标题():
    """长文的子块靠这个前缀保住主题，不至于脱离上下文。"""
    d = _by_title(splitter.split_documents(SAMPLE), "普通文章")[0]
    assert d.page_content.startswith("分类甲 > 小节一 > 普通文章")


def test_只有链接的文章仍然入库_只嵌入标题():
    """正文缺失的文章不能整篇丢掉，否则用户问到时连链接都推不出来。"""
    docs = _by_title(splitter.split_documents(SAMPLE), "只有链接没有正文的文章")
    assert len(docs) == 1
    assert docs[0].metadata["url"].endswith("222-linkonly")
    assert docs[0].page_content.strip()  # 不能是空的


def test_没有链接的文章url为空字符串():
    d = _by_title(splitter.split_documents(SAMPLE), "没有链接的文章")[0]
    assert d.metadata["url"] == ""


def test_绝不产出空块():
    """空字符串送去 embedding 会返回 2560 维而非 1024，写入 Chroma 会维度冲突。"""
    docs = splitter.split_documents(SAMPLE + "\n### 空文章\n\n\n")
    assert all(d.page_content.strip() for d in docs)


def test_短文章不被二次切分():
    """文章长度中位数 1053 落在 800 和 1500 之间，不能对所有块统一按 800 切。"""
    body = "短" * 1000  # 1000 字，未超过 1500 阈值
    docs = splitter.split_documents(f"# 甲\n\n## 乙\n\n### 中等长度文章\n\n{body}\n")
    assert len(docs) == 1
    assert docs[0].metadata["chunk_total"] == 1


def test_长文章被二次切分且共享同一份元数据():
    body = "这是一段很长的正文内容。" * 300  # 远超 1500 字
    docs = splitter.split_documents(f"# 甲\n\n## 乙\n\n### 超长文章\n\n{body}\n")
    assert len(docs) > 1
    assert {d.metadata["title"] for d in docs} == {"超长文章"}
    assert all(d.metadata["chunk_total"] == len(docs) for d in docs)
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
