# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
# kubernetes(82MB) 和 onnxruntime(58MB) 是 chromadb 拉进来的：前者服务于它的
# 集群部署模式，后者是它默认的 ONNX 嵌入模型。本项目用的是本地持久化模式 +
# 百炼嵌入，两个都用不到，卸掉省约 140MB。已实测卸载后检索结果完全一致。
# 必须与 pip install 放在同一个 RUN 层——Docker 分层是累加的，
# 在后续层里卸载并不会让镜像变小。
# 注意：grpcio 不能卸，chromadb 的 telemetry 模块会顶层 import 它。
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y kubernetes onnxruntime

# 复制应用代码
COPY config.py llm.py app.py ui.py ingest.py calibrate.py ./
COPY rag/ ./rag/

# 预先建好的向量索引。索引是 git 里的产物，不在构建时生成——
# 构建时生成需要把 API key 传进构建层，且每次部署都要重跑几十次嵌入调用。
COPY vector_store/ ./vector_store/

# 知识库原文。仅供容器内重跑 ingest.py 时使用，服务运行本身不读它。
COPY all-articles.md .

EXPOSE 8000

# 用 ${PORT:-8000}: Railway 等平台会注入 $PORT 要求应用监听它；
# 本地不设 PORT 时仍然是 8000，行为不变。
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
