# allen-agent

基于 FastAPI + 阿里云百炼（DashScope）的 Agent 服务，带 Ballet 支持文档的 RAG 知识库检索和 Gradio 网页入口。

## 它做什么

问 Ballet 冷钱包相关问题时，自动检索官方支持文档并附上原文链接；文档里没有的问题，
直接由大模型作答。**路由是自动的**，调用方不需要传任何开关。

## 接口

| 方法 | 路径      | 说明                                    |
|------|-----------|-----------------------------------------|
| GET  | `/ui`     | **Gradio 聊天页面**（与接口同进程同端口）|
| POST | `/chat`   | 核心 Agent 接口，自动判定走文档还是走模型 |
| GET  | `/health` | 健康检查                                |
| GET  | `/myapp`  | 示例接口                                |
| GET  | `/docs`   | Swagger 交互式文档                      |

`/chat` 的响应体：

```json
{
  "reply": "回答正文",
  "mode": "rag",
  "sources": [
    {"title": "文章标题", "url": "https://support.ballet.com/...", "category": "分类", "score": 0.8276}
  ]
}
```

`mode` 为 `rag` 表示基于知识库作答（`sources` 非空）；为 `llm` 表示知识库里没有相关文档，
由模型自身知识作答（`sources` 为空）。

## 本地运行

```bash
uv venv .venv                        # 或 python3 -m venv .venv
uv pip install -r requirements.txt   # 或 .venv/bin/pip install -r requirements.txt
cp .env.example .env                 # 填入真实的 DASHSCOPE_API_KEY
.venv/bin/python ingest.py           # 建立向量索引（约 18 秒，仅需一次）
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

打开 <http://127.0.0.1:8000/ui> 即可对话。命令行测试：

```bash
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"Ballet 有客服电话吗？"}'
```

## 知识库与 RAG

知识库是项目根目录的 `all-articles.md`（Ballet 支持中心全部文章，194 篇）。

### 切块

利用文档天然的三级结构：`#` 分类 / `##` 小节 / `###` 文章。

- 按标题切出 194 篇，元数据带 `category` / `section` / `title`
- 抽取每篇正文里的原文链接存进 `url`，回答时用来附引用
- 仅对 **>1500 字符**的长文用 `RecursiveCharacterTextSplitter(800, overlap=120)` 二次切分。
  文章长度中位数是 1053，若对所有块统一按 800 切会把普通文章无谓劈成两半
- 每个块嵌入前前置 `分类 > 小节 > 标题`，避免长文中段的子块脱离主题
- 最终 **315 个块**，覆盖 194/194 篇

### 检索与路由

```
提问 → Chroma 检索 top-4 → top1 分数 ≥ 0.45 ？
                              ├─ 是 → 带文档回答，mode="rag"
                              └─ 否 → 纯大模型回答，mode="llm"
```

提示词里另有一道兜底：即使分数过阈但资料实际答不了，模型会说明"文档中未涵盖"
并改用自身知识，而不是从无关资料里硬凑。

### 阈值怎么来的

**实测标定，不是拍脑袋。** 执行 `python calibrate.py`，它用 15 个域内问题
（中英混合）和 10 个域外问题跑一遍并打印分数分布：

```
域内最低分: 0.5096      域外最高分: 0.3853
✅ 两组完全可分，间隔 0.1244  →  建议 RAG_SCORE_THRESHOLD = 0.45
```

跨语言检索有效：中文提问 "Ballet 有客服电话吗？" 能以 0.8442 的分数命中英文文章
*Does Ballet provide phone service?*。

**改文档或换嵌入模型后，请重跑 `ingest.py` 和 `calibrate.py`，不要凭感觉调阈值。**

## Docker

```bash
python ingest.py                   # 构建镜像前必须先有 vector_store/
docker build -t allen-agent .
docker run -d --name allen-agent -p 8000:8000 --env-file .env allen-agent
```

常用操作：

```bash
docker logs -f allen-agent     # 看日志
docker stop allen-agent        # 停止
docker start allen-agent       # 再次启动
docker rm -f allen-agent       # 删除容器
```

镜像里不含 `.env`（`.dockerignore` 已排除），密钥通过 `--env-file` 或 `-e` 注入。
`load_dotenv` 不会覆盖已存在的环境变量，所以这种方式可以正常工作。

## 部署到 Railway

Railway 的 GitHub 集成默认开启自动部署，push 到 `main` 即自动构建，无需手动触发。

**需要在 Railway 的环境变量面板中配置**（`.env` 不进仓库也不进镜像）：

```
DASHSCOPE_API_KEY=<你的密钥>
DASHSCOPE_BASE_URL=https://<你的工作空间>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.7-plus
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
GRADIO_ANALYTICS_ENABLED=False
```

`PORT` 由 Railway 自动注入，不用手动配。

## 注意事项

- **`vector_store/` 必须提交进 git。** Railway 是从 GitHub 拉代码构建的，索引不在仓库里
  就进不了镜像。届时容器会**静默降级**成纯大模型模式——健康检查全绿、接口全通、页面能聊，
  但 RAG 完全失效，极难发现。同理，它也**不能**加进 `.dockerignore`。
  CI 里有一道检查专门卡这个（`docker logs | grep 'RAG 生效'`）。
- **嵌入批量上限是 10 条。** 百炼超过 10 条会返回
  `400 InternalError.Algo.InvalidParameter: batch size is invalid`，
  所以 `OpenAIEmbeddings` 必须设 `chunk_size=10`。
- **`check_embedding_ctx_length` 必须设为 `False`。** 默认 `True` 时 LangChain 会把文本
  转成 token 数组再发送，而百炼只接受字符串数组，会返回
  `400 input must be an array of strings`。
- **空文本不能送去嵌入。** 百炼对空字符串返回 **2560 维**（而非正常的 1024 维），
  混进去会让 Chroma 报维度冲突。切块器里已做过滤。
- **不要用 `text-embedding-v3`。** 该端点上它虽然能调通，但不在 `/models` 列表里，
  属于未公开别名，随时可能失效。用列表中正式挂载的 `qwen3.7-text-embedding`。
- **代理**：本机全局 `HTTPS_PROXY` 会让到模型服务的 TLS 握手失败，因此 `llm.py` 里给
  OpenAI 客户端传了 `httpx.Client(trust_env=False)` 直连。若你的网络必须走代理，改成 `True`。
- **Gradio 页面不猜端口。** 页面通过 `httpx.ASGITransport` 把请求直接投递给 app 对象，
  完整经过 `/chat` 的路由、pydantic 校验和响应序列化，但不经过 TCP。
  早先的写法是 POST 到 `http://127.0.0.1:{PORT}/chat`，靠 `PORT` 环境变量猜端口——而真实
  端口来自 uvicorn 的 `--port`。两者不一致时（例如 `uvicorn --port 8123` 却没设 `PORT`），
  自调用会静默打到 8000 上**别的**服务；实测曾打进一条 SSH 隧道并返回了毫不相干的结果，
  页面看着正常但 RAG 全部失效。`tests/test_routing.py` 里有回归测试锁住这一点。
- **密钥**：`.env` 已在 `.gitignore` 中，不会被提交。

## 启用 CI

CI 配置在 `.github/ci.yml.disabled`，暂时没放在 `.github/workflows/` 下——
推送时的 OAuth token 缺少 `workflow` scope，GitHub 会拒绝任何创建/修改
`.github/workflows/` 下文件的推送。

补上 scope 后启用：

```bash
gh auth refresh -h github.com -s workflow   # 需完成浏览器授权
mkdir -p .github/workflows
git mv .github/ci.yml.disabled .github/workflows/ci.yml
git commit -am "enable CI" && git push
```

这套 CI 除了跑单元测试和构建镜像，还有一道检查专门卡"RAG 静默降级"：
容器起来后断言日志里出现 `RAG 生效`，否则构建失败。
