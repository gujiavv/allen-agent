# allen-agent：Gradio 前端入口 + LangChain RAG 知识库检索

- 日期：2026-08-29
- 状态：设计已确认，待编写实施计划

## 1. 背景与现状

`allen-agent` 目前是一个最小 FastAPI 服务：

- 单文件 `app.py`（约 60 行），三个端点 `/chat`（POST，调大模型）、`/health`、`/myapp`
- 通过 OpenAI SDK 调 DeepSeek，`openai==1.6.1`、`httpx==0.27.2`（1.6.1 不兼容 httpx≥0.28）
- 因本机全局 `HTTPS_PROXY` 到 `api.deepseek.com` 的 TLS 握手失败，客户端传了 `httpx.Client(trust_env=False)` 直连
- 6 个 pytest 测试（全部 mock，不联网）、Dockerfile（python:3.13-slim，242MB）、GitHub Actions CI
- 项目根目录有 `all-articles.md`：Ballet 冷钱包支持中心全部文章，尚未被任何代码使用

## 2. 目标

1. 提供一个网页聊天入口，用 Python 框架实现，通过 HTTP 调用现有 `/chat` 接口
2. 基于 `all-articles.md` 做 LangChain RAG：文档切块 + 向量数据库 + 检索
3. **自动路由**：能从文档检索到就用文档回答，检索不到就退回大模型自身知识

## 3. 知识库文档特征（已实测）

`all-articles.md`，259KB / 3759 行，Ballet 冷钱包支持文档：

| 维度 | 数值 |
|---|---|
| 语言 | 英文 99.86%（中文字符仅 254 个） |
| 一级标题 `#`（分类） | 13 个 |
| 二级标题 `##`（小节） | 45 个 |
| 三级标题 `###`（文章） | 194 篇 |
| 单篇长度 | 最短 247，中位数 1053，平均 1296，最长 4940 字符 |
| 长度分布 | P75=1642，P90=2622，P95=3286 |
| 超长文章 | >1500 字符 53 篇，>3000 字符 11 篇 |

每篇文章的 `###` 标题下一行都是该文章的原文链接，形如
`<https://support.ballet.com/hc/en-us/articles/...>`。**这是可直接利用的引用来源。**

**跨语言检索是硬需求**：知识库为英文，用户以中文提问，embedding 模型必须支持中英跨语言召回。

## 4. 模型服务：迁移到阿里云百炼

项目从 DeepSeek 整体切换到百炼（DashScope）专属工作空间端点。

```
DASHSCOPE_BASE_URL=https://ws-dgldt8xq9prfuhug.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.7-plus
DASHSCOPE_MODEL_TURBO=qwen-turbo
```

> 关键前提：DeepSeek 官方不提供 embedding 接口，只有 chat/completions。做 RAG 必须另找 embedding 来源，这是切换到百炼的直接原因。

### 4.1 端点实测结论

对该专属端点做过实际探测，结论如下：

| 探测项 | 结果 | 对设计的影响 |
|---|---|---|
| `/models` 返回模型数 | 246 个 | — |
| `qwen3.7-plus` chat | ✅ 可用 | 作为主对话模型 |
| `qwen-turbo` | ✅ 在列表中 | 备用轻量模型 |
| `qwen3.7-text-embedding` | ✅ 可用，**1024 维** | **选定为 embedding 模型** |
| `text-embedding-v3` | ⚠️ 能调通且为 1024 维，但**不在 `/models` 列表中** | 未公开别名，可能随时失效，**不采用** |
| 批量 embedding 上限 | **一次最多 10 条**，超过返回 400 `batch size is invalid` | **必须设 `chunk_size=10`**，否则建索引直接失败 |
| `encoding_format=base64` | ✅ 支持 | 可直接用 `langchain-openai` 的 `OpenAIEmbeddings`，无需引入 `dashscope` SDK |
| `encoding_format=float` | ✅ 支持，返回 1024 维 list | — |
| `dimensions=512` | ✅ 支持降维 | 保留优化余地，默认仍用 1024 |
| **空字符串输入** | ⚠️ **返回 2560 维**（非 1024） | **切块后必须过滤空白块**，否则写入 Chroma 时维度冲突报错 |

### 4.2 密钥安全

用户提供的 `DASHSCOPE_API_KEY` 曾以明文出现在对话记录中。

- **行动项：实施完成后到百炼控制台吊销该 key 并重新生成**
- `.env` 已在 `.gitignore` 中，不会进入 git
- `.env.example` 只放占位符，不放真实值

## 5. 技术选型与理由

| 决策点 | 选择 | 理由 | 被否方案 |
|---|---|---|---|
| Embedding | 百炼 `qwen3.7-text-embedding` | 国内直连不用代理；跨语言支持；端点正式挂载 | 本地 HuggingFace 模型（要装 torch，镜像 242MB→2.5GB）；OpenAI（需另一个 key 且可能要走代理） |
| 向量库 | **Chroma 持久化到本地目录** | LangChain 一等公民，资料多；支持按元数据过滤（正好配三级标题结构）；落盘后重启秒加载不重复花钱；单容器部署不引入外部依赖 | FAISS（元数据过滤弱、pickle 反序列化）；pgvector（用户的 Postgres 是 medusa 电商项目在用，且 localhost 库 Docker 连不到，部署到 Railway 还要另开托管 PG） |
| 前端 | **Gradio 挂载到 FastAPI** | `gr.ChatInterface` 约 40 行搞定聊天气泡/历史/流式；`gr.mount_gradio_app` 同进程同端口，Dockerfile 和 CI 几乎不动 | Streamlit（必须独立跑 8501 端口，Docker 要双进程或拆容器）；Jinja2+原生 JS（要手写 100+ 行 JS，不算"用 Python 写前端"）；NiceGUI（小众，踩坑资料少） |
| 路由判定 | **相似度阈值主判 + 提示词兜底** | 快、便宜、行为确定、可单测；闲聊问题不浪费大提示词 | 全交给大模型判断（每次闲聊也要花 embedding 和大提示词 token，走了哪条路不可控、难测）；qwen-turbo 意图路由（每个问题多 0.5~1s 往返，多一个失败点） |

## 6. 模块结构

`app.py` 目前是 60 行单文件，加入 RAG 和前端后必须拆分。按"一个文件一个职责"：

```
app.py            FastAPI 装配：路由定义 + 挂载 Gradio（保持薄）
config.py         环境变量集中读取与校验
llm.py            百炼客户端 + chat 调用封装
rag/__init__.py
rag/splitter.py   Markdown 切块 + 元数据抽取（分类/小节/标题/原文 URL）
rag/store.py      Chroma 建库 / 加载 / 带分数检索
rag/pipeline.py   检索 → 阈值判定 → 组装提示词 → 产出回答与引用
ui.py             Gradio ChatInterface
ingest.py         建索引 CLI（一次性执行）
calibrate.py      阈值标定 CLI
vector_store/     Chroma 落盘目录（加入 .gitignore）
```

边界约定：`rag/` 三个模块不互相依赖对方内部实现。`splitter` 是纯函数、不联网；`store` 只负责存取；`pipeline` 负责编排。因此切块逻辑与路由判定逻辑都能脱网单元测试。

## 7. 切块设计

利用文档天然的三级标题结构：

1. **第一刀**：`MarkdownHeaderTextSplitter` 按 `#` / `##` / `###` 切分 → 194 块，每块携带元数据 `{category, section, title}`
2. **抽取 URL**：从每篇正文中提取 `<https://support.ballet.com/...>` 行，存入元数据 `url` 字段，使回答可附真实原文链接
3. **第二刀**：**仅对长度 >1500 字符的块**（53 篇）用 `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)` 二次切分；**≤1500 字符的块原样保留，不切**。

   这条规则必须按字面实现，不可简化为"对所有块统一跑 chunk_size=800"——那样会把中位数 1053 字符的普通文章也劈成两半，破坏完整语境。文章长度中位数正好落在 800 和 1500 之间，两种做法结果差异很大。
4. **上下文增强**：每个子块在送去 embedding 前，前置 `分类 > 小节 > 标题`，避免长文章中段的子块脱离主题。注意——前置内容只影响被嵌入的文本，不改变展示给用户的原文
5. **过滤空块**：丢弃 strip 后为空的块（见 4.1 空字符串返回 2560 维的问题）

预计产出约 280 块；按批量上限 10 条计算，全量建索引约 28 次 API 调用。

## 8. 自动路由设计

```
用户提问
  ↓
Chroma similarity_search_with_relevance_scores(query, k=4)
  ↓
top1 分数 ≥ RAG_SCORE_THRESHOLD ?
  ├─ 是 → 拼接检索到的上下文 → qwen3.7-plus → mode="rag"，附 sources
  └─ 否 → 直接提问 qwen3.7-plus       → mode="llm"，sources 为空
```

**第二道保险（提示词兜底）**：即使分数过阈但内容实际答不了该问题，系统提示词要求模型改用自身知识作答，并明确说明"该内容未被文档覆盖"，而不是从无关上下文里硬编答案。

**阈值标定，不拍脑袋定值**：`calibrate.py` 用两组问题跑一遍并打印分数分布——

- 域内问题 15 个（从真实文章标题改写而来，中英各有）
- 域外问题 10 个（闲聊、常识、其他领域）

取能分开两组分布的值作为 `RAG_SCORE_THRESHOLD` 默认值写入 `.env.example` 与 `config.py`。标定结果需记录在 README 中。

## 9. 接口契约

`/chat` 保持单一端点（前端调的就是它），响应体扩展：

```python
class Source(BaseModel):
    title: str
    url: str
    category: str
    score: float

class ChatResponse(BaseModel):
    reply: str
    mode: Literal["rag", "llm"]     # 走了哪条路，前端据此显示徽标
    sources: list[Source] = []      # 仅 rag 模式下非空
```

`ChatRequest` 不变，仍只有 `message` 字段——路由由服务端自动决定，不暴露开关。

**对现有测试的影响（已获用户确认）**：`test_chat_success` 与 `test_chat_passes_model_and_message` 使用 `r.json() == {"reply": ...}` 全等断言，响应体加字段后会失败，需改为断言 `r.json()["reply"]`。其余 4 个测试不受影响。

## 10. 索引生命周期

- `ingest.py` **手动执行一次**，读 `all-articles.md`，生成 `vector_store/`
- **启动时不自动建索引**。否则每次冷启动都要 28 次 API 调用，既拖慢启动又在无人察觉时消耗额度
- 启动时行为：`vector_store/` 存在 → 加载；不存在 → 打印警告并降级为纯 LLM 模式，服务照常启动，不崩溃
- Docker：`COPY vector_store/ ./vector_store/` 把索引烤进镜像。构建前必须先在本地跑过 `ingest.py`，README 需写明此前置步骤

  ⚠️ **`vector_store/` 要加进 `.gitignore`，但绝不能加进 `.dockerignore`**。两者是不同机制：git 忽略它是因为索引是二进制产物不该进版本库；而 Docker 构建上下文读的是本地文件系统而非 git，一旦误加进 `.dockerignore`，镜像里就没有索引，容器启动后会静默降级成纯 LLM 模式——服务能起、接口能通，但 RAG 完全失效，很难察觉。当前 `.dockerignore` 内容为 `.env / .venv/ / venv/ / __pycache__/ / .git/ / .gitignore / README.md`，不含 `vector_store/`，符合要求，改动时勿动

## 11. 前端设计

- `gr.ChatInterface` 通过 `gr.mount_gradio_app(app, demo, path="/ui")` 挂载到现有 FastAPI 实例，同进程同端口
- handler 发一个真实的 HTTP 请求打到本服务的 `/chat`，完整经过路由、pydantic 校验与响应序列化

  **权衡**：UI 走的代码路径与外部调用者完全一致，不会出现"页面能用但接口是坏的"这类偏差。已获用户确认。

  **实施期修订**：原计划 `httpx.post("http://127.0.0.1:${PORT}/chat")`，实测发现严重缺陷——
  `PORT` 环境变量与 uvicorn 实际的 `--port` 参数是两个独立来源，不一致时自调用会静默打到
  另一个端口上的**其他服务**（本地实测打进了 8000 端口的一条 SSH 隧道，返回了不相关的结果，
  页面显示正常但 RAG 全部失效，且存在把用户提问发给第三方的风险）。
  改为 `httpx.ASGITransport(app=app)`：请求仍完整走 `/chat` 的 ASGI 链路，但不猜端口、不经 TCP。
  `config.PORT` 随之删除。回归测试见 `tests/test_routing.py::test_页面自调用不依赖端口`。
- 回答下方展示 `mode` 徽标（📚 文档 / 🤖 模型）与可点击的原文链接列表
- 错误以聊天气泡形式提示，不向用户抛 traceback

## 12. 依赖变更

新增：`langchain`、`langchain-community`、`langchain-chroma`、`chromadb`、`langchain-openai`、`gradio`

**`openai==1.6.1` 必须升级**：`langchain-openai` 要求较新版本的 openai SDK。升级后，`httpx==0.27.2` 的锁定理由（1.6.1 不兼容 httpx≥0.28 的 `proxies` 参数移除）随之解除，httpx 可一并放开。

升级时必须保留 `trust_env=False` 的绕代理处理——本机全局代理会导致 TLS 握手失败，这个约束在切换到百炼后依然存在，需实测确认。

镜像体积预计从 242MB 增至约 450MB。

## 13. 错误处理

| 故障场景 | 处理方式 |
|---|---|
| `vector_store/` 不存在 | 启动打警告，降级为纯 LLM 模式，服务正常起 |
| 查询时 embedding 接口报错 | 捕获后降级走纯 LLM 路径，`mode="llm"`，不让整个请求失败 |
| chat 模型调用失败 | 维持现有行为，转为 HTTP 500（现有测试已覆盖） |
| 请求体缺 message | 维持现有行为，pydantic 返回 422（现有测试已覆盖） |
| Gradio 端异常 | 在聊天气泡内展示友好错误信息 |

## 14. 测试方案

全部 mock，不联网，CI 保持绿色。

- **改造**：`test_chat_success`、`test_chat_passes_model_and_message` 适配新响应体
- **切块测试**（纯函数，无网络）：元数据三级标题正确、URL 被正确抽取、超长文章被二次切分、空白块被过滤、上下文增强不污染展示原文
- **路由测试**（mock 检索器）：分数高于阈值 → `mode="rag"` 且 sources 非空；低于阈值 → `mode="llm"` 且 sources 为空；embedding 抛异常 → 降级为 `mode="llm"` 而非 500
- **前端**：`GET /ui` 返回 200
- **保留**：现有 `/health`、`/myapp`、上游报错转 500、畸形请求体 422 四个测试

## 15. 不做的事（YAGNI）

- 多轮对话历史送入检索（首版单轮提问检索，Gradio 自身保留对话展示）
- 重排序模型（rerank）
- 流式输出（先跑通，后续可加）
- 用户认证 / 多租户
- 增量索引更新（文档变了就重跑 `ingest.py`）
- 检索评估指标体系（`calibrate.py` 打印分布已足够定阈值）

## 16. 风险

| 风险 | 应对 |
|---|---|
| 阈值标定后仍误判（中文问句召回英文文档分数偏低） | 提示词兜底是第二道保险；必要时降低阈值并依赖模型判断 |
| 专属端点模型列表未来变动 | 已避开未公开别名 `text-embedding-v3`，只用列表中的 `qwen3.7-text-embedding` |
| openai SDK 升级破坏现有代理绕行方案 | 升级后立即实测 `/chat` 连通性，保留 `trust_env=False` |
| Gradio 与 FastAPI 版本兼容（当前 fastapi==0.104.1 较旧） | 实施时验证，必要时同步升级 fastapi |
