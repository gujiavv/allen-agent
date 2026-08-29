# ingest.py
"""建立向量索引。手动执行一次即可：python ingest.py

刻意不在服务启动时自动建索引——那样每次冷启动都要几十次 embedding 调用，
既拖慢启动又在无人察觉时消耗额度。
"""
import sys

import config
from rag import splitter, store


def main() -> int:
    if not config.KNOWLEDGE_FILE.exists():
        print(f"❌ 找不到知识库文件：{config.KNOWLEDGE_FILE}")
        return 1

    print(f"读取 {config.KNOWLEDGE_FILE.name} ...")
    documents = splitter.load_and_split(config.KNOWLEDGE_FILE)

    articles = len({d.metadata.get("title") for d in documents})
    print(f"切分完成：{articles} 篇文章 → {len(documents)} 个块")
    print(f"开始嵌入（模型 {config.EMBEDDING_MODEL}，每批 {config.EMBEDDING_BATCH_SIZE} 条）...")

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total}", end="", flush=True)

    store.build(documents, on_progress=progress)
    print(f"\n✅ 索引已写入 {config.VECTOR_STORE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
