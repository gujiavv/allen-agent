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

## 流式输出与思考过程

页面上的回答是逐字流出来的，模型的推理过程收在一个默认折叠的 **💭 思考** 面板里，
点开可以看到完整推理，面板上标着耗时。

- **接口**：`POST /chat/stream`，SSE。事件类型依次为
  `mode` → `sources` → `thinking`(多条) → `content`(多条) → `done`。
  引用在正文之前就推送，所以页面能先显示"找到了哪几篇文档"再开始出字。
- **`/chat` 保持非流式**，响应契约一字未变。流式单开一个端点而不是给 `/chat` 加
  `stream` 参数——同一个端点返回两种形态会让契约变含混，外部调用方也会被波及。
- **不需要 `enable_thinking` 参数**。qwen3.7-plus 默认就在 `reasoning_content` 里
  返回推理过程。`reasoning_content` 不是 OpenAI 官方字段，代码里用 `getattr` 取。
- **推理语言跟着系统提示词走**：走 RAG 时系统提示词是中文，推理也是中文；
  纯大模型路径没有系统提示词，推理可能是英文。这是模型行为，提示词左右不了。
- 响应头带了 `X-Accel-Buffering: no`，防止反向代理把流式攒成一坨再发。

## 关于 vector_store 老是"被修改"

本地跑一次服务或测试之后，`git status` 会显示这两个文件被改动：

```
 M vector_store/<uuid>/data_level0.bin
 M vector_store/chroma.sqlite3
```

**这是正常的，索引内容并没有变。** Chroma 打开数据库时会改写内部簿记
（SQLite 的页头、HNSW 的运行时状态），哪怕只是只读查询也会。已验证过两个版本的
块数都是 315、文件大小一字节不差。

直接丢弃即可：

```bash
git checkout -- vector_store/
```

**只有 `all-articles.md` 变了才需要重建索引并提交：**

```bash
.venv/bin/python ingest.py
.venv/bin/python calibrate.py    # 语料变了，阈值要重新标定
git add vector_store/ && git commit -m "rebuild vector index"
```

注意重建后一定要重跑 `calibrate.py`：阈值 0.45 是针对当前这批文档实测出来的，
换了语料就不一定还能把域内/域外问题分开。

## 怎么确认 Railway 部署的是最新代码

`/health` 会自报版本，不用去翻部署面板：

```bash
curl -s https://<你的域名>/health
```

```json
{
  "status": "ok",
  "version": "2.1",
  "commit": "17a76ed",     ← 跟 git rev-parse --short HEAD 对比
  "branch": "main",
  "rag_enabled": true,     ← false 说明索引没进镜像，RAG 是死的
  "chunks": 315
}
```

对照本地：`git rev-parse --short HEAD`。两者一致就说明线上是最新代码。

### commit 显示 unknown 怎么办

说明 Railway 没有注入 `RAILWAY_GIT_COMMIT_SHA`，**该服务不是从 GitHub 仓库构建的**
——多半是当初用 `railway up` 从本地上传创建的。这种服务和 GitHub 没有任何关联，
`git push` 再多次也不会触发部署。

去 Railway：**Settings → Source → Connect Repo**，接上 `gujiavv/allen-agent` 的
`main` 分支。接上之后每次 push 才会自动构建。

### commit 是旧的怎么办

自动部署没触发或构建失败了。依次检查：

1. **Settings → Source** 里的分支是不是 `main`
2. 同一页的 **Auto Deploy** 有没有被关掉
3. **Deployments** 页面看最近一次构建是成功还是失败；失败就看 Build Logs

手动补救：Deployments 页面右上角 **Deploy / Redeploy**，会拉当前分支最新提交重建。

### 部署成功但 rag_enabled 是 false

索引没进镜像。确认 `vector_store/` 确实在仓库里（`git ls-files vector_store/ | head`），
并且没被写进 `.dockerignore`。这是本项目最容易翻车的地方：索引缺失时服务照常启动、
`/health` 照常返回 ok，只有 RAG 是死的。

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
