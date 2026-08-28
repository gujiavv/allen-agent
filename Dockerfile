# Dockerfile
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动命令
# 用 ${PORT:-8000}: Railway 等平台会注入 $PORT 要求应用监听它；
# 本地不设 PORT 时仍然是 8000，行为不变。
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
