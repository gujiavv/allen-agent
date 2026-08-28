# allen-agent

基于 FastAPI + DeepSeek 的最小 Agent 服务。

## 接口

| 方法 | 路径      | 说明                          |
|------|-----------|-------------------------------|
| POST | `/chat`   | 核心 Agent 接口，调用大模型   |
| GET  | `/health` | 健康检查                      |
| GET  | `/myapp`  | 示例接口                      |
| GET  | `/docs`   | Swagger 交互式文档            |

## 本地运行

```bash
uv venv .venv                        # 或 python3 -m venv .venv
uv pip install -r requirements.txt   # 或 .venv/bin/pip install -r requirements.txt
cp .env.example .env                 # 填入真实的 DEEPSEEK_API_KEY
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

测试：

```bash
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

## Docker

```bash
docker build -t allen-agent .
docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx \
  -e DEEPSEEK_BASE_URL=https://api.deepseek.com/v1 \
  -e DEEPSEEK_MODEL=deepseek-v4-flash allen-agent
```

镜像里不含 `.env`（`.dockerignore` 已排除），密钥通过 `-e` 注入。
`load_dotenv` 不会覆盖已存在的环境变量，所以这种方式可以正常工作。

## 注意事项

- **代理**：本机全局 `HTTPS_PROXY` 到 `api.deepseek.com` 的 TLS 握手会失败，
  因此 `app.py` 里给 OpenAI 客户端传了 `httpx.Client(trust_env=False)` 直连。
  若你的网络必须走代理，把它改成 `True`。
- **httpx 版本**：`openai==1.6.1` 不兼容 `httpx>=0.28`（`proxies` 参数被移除），
  requirements.txt 中已锁定 `httpx==0.27.2`。
- **密钥**：`.env` 已在 `.gitignore` 中，不会被提交。
