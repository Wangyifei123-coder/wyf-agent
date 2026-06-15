# Session Handoff

## 当前已验证

- 项目骨架已创建，7 层架构完整
- 57/57 测试通过
- ruff check 全部通过
- mypy 全部通过
- Agentic RAG 全流程验证通过
- 三路混合检索验证通过（向量 + BM25 + Neo4j 知识图谱）
- Neo4j Docker 部署并验证连接
- 评估数据集 41 组，评估得分 0.88
- Streamlit 前端对话界面完成（登录、流式输出、意图标签、引用来源）
- Docker 一键部署配置完成
- JWT 认证鉴权完成
- Prometheus 监控指标完成
- 增量更新支持完成
- LLM 真流式输出完成
- Embedding 模型预热完成
- **端到端对话测试通过**（2026-06-15）
  - 健康检查 ✅
  - JWT 登录 ✅
  - 文档入库（7 文档，7 分块）✅
  - 知识问答（关键词 100% 命中）✅
  - 流式输出（超时问题待优化）⚠️
- **多模态图片支持**（2026-06-15）
  - 模型升级到 mimo-v2.5 ✅
  - API 支持 images 字段 ✅
  - 前端图片上传/预览 ✅
  - 图片入库（LLM 描述）✅
  - 端到端多模态测试通过 ✅
    - 文字识别：正确识别图片中的文字
    - 形状描述：正确描述图片中的图形
- **网页入库功能**（2026-06-15）
  - 输入 URL 提取文字内容（trafilatura）✅
  - 提取并下载网页图片 ✅
  - 图片用 LLM 生成描述后入库 ✅
  - API 端点 `/knowledge/ingest-url` ✅
  - 聊天框直接粘贴 URL 自动入库 ✅
- **黄色优先级任务完成**（2026-06-15）
  - 评估数据集扩充：41 → 109 组 ✅
  - 文档入库测试：6 种格式全部通过 ✅
  - 前端样式优化：现代化 CSS 样式 ✅
  - 流式输出优化：超时/重试/快速路径 ✅
- **工具调用能力**（2026-06-15）
  - MCP 协议支持 ✅
  - 工具发现和注册 ✅
  - LLM 意图识别工具调用 ✅
  - 工具执行和结果返回 ✅
  - 工具调用审计日志 ✅
  - JSON-RPC 2.0 统一消息格式 ✅
  - MCP Registry 自动发现和安装 ✅
  - 5 个 MCP 服务器（test-tools, filesystem, database, search, api）✅
  - 规则匹配 + 置信度判断 ✅
  - OpenAI 原生 Function Calling ✅
  - 输入校验（JSON Schema）✅
  - 超时控制（asyncio.wait_for）✅
  - 熔断机制（CircuitBreaker）✅
  - 结果截断（max_result_length）✅
  - 审计日志（入参/出参/耗时）✅
  - 并行工具调用（asyncio.gather）✅
  - 工具结果缓存（LRU + TTL）✅
  - 工具调用链（多步骤执行）✅
  - 流式工具进度（SSE）✅
  - 工具版本管理（major.minor.patch）✅
  - 工具权限控制（角色权限）✅
  - 工具调用重试（指数退避）✅

## 本轮改动

- 实现 Agentic RAG 全流程（LangGraph 状态机）
- 实现三路混合检索（向量 + BM25 + Neo4j 知识图谱，RRF 融合）
- 创建评估数据集（41 组测试用例）+ 评估脚本
- 修复 Tool Call 泄露（/chat 直接走 RAGGraph）
- 优化查询改写（代词消解）和问题分解（多文档检索）
- 创建 Streamlit 前端对话界面（登录、流式输出、文档上传）
- 创建 Docker 部署配置（docker-compose）
- 实现 JWT 认证鉴权
- 实现 Prometheus 监控指标
- 实现增量更新（delete_by_source、rebuild、sources 列表）
- 实现 LLM 真流式输出（SSE）
- 实现 Embedding 模型预热
- 支持 .docx、.xlsx、.csv 文档格式
- 修复 emoji 编码问题
- 修复 intent Enum 显示问题

## 新增文件

```
src/rag/
├── __init__.py
├── loader.py           # 文档加载（Markdown/PDF/TXT/DOCX/XLSX/CSV）
├── splitter.py         # 结构感知文本分块
├── embeddings.py       # text2vec 中文 Embedding（HF 镜像）
├── vectorstore.py      # ChromaDB 封装（增量更新）
├── retriever.py        # RRF 重排序检索
├── bm25_retriever.py   # BM25 关键词检索
├── kg_retriever.py     # Neo4j 知识图谱检索
├── hybrid_retriever.py # 三路 RRF 融合
└── graph.py            # LangGraph Agentic RAG 状态机（流式输出）
src/tools/
└── knowledge.py        # 知识检索工具
src/safety/
└── auth.py             # JWT 认证模块
src/observability/
└── metrics.py          # Prometheus 指标
evals/
├── test_dataset.json   # 41 组评估用例
├── run_eval.py         # 评估脚本
└── eval_results.json   # 评估结果
frontend.py             # Streamlit 前端（登录、流式输出）
docker/
├── Dockerfile          # 后端镜像
├── Dockerfile.frontend # 前端镜像
└── docker-compose.yaml # 一键部署
```

## 启动命令

```bash
# 进入项目
cd D:\PythonProject\my-agent\wyf-agent

# 安装
pip install -e ".[dev]"

# 验证
pytest tests/ -v
python -m ruff check src/
python -m mypy src/

# 启动后端
python -m uvicorn src.api:app --port 8080

# 启动前端
python -m streamlit run frontend.py --server.port 8501

# Docker 一键部署
docker compose -f docker/docker-compose.yaml up -d

# 停止服务
netstat -ano | findstr ":8080" | findstr "LISTENING"
taskkill /PID <pid> /F
```

## API 端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/auth/login` | POST | 否 | JWT 登录 |
| `/auth/verify` | GET | 否 | 验证 token |
| `/chat` | POST | 是 | 对话（非流式） |
| `/chat/stream` | POST | 是 | 对话（SSE 流式） |
| `/knowledge/ingest` | POST | 否 | 文档入库（增量） |
| `/knowledge/rebuild` | POST | 否 | 全量重建 |
| `/knowledge/stats` | GET | 否 | 知识库状态 |
| `/knowledge/sources` | GET | 否 | 已入库文件列表 |
| `/prometheus` | GET | 否 | Prometheus 指标 |
| `/health` | GET | 否 | 健康检查 |

## 配置

- API Key: `config/.env`（已 gitignore）
- 模型: `anthropic/mimo-v2.5-pro`
- 端点: `https://token-plan-cn.xiaomimimo.com/anthropic`
- 后端端口: `8080`
- 前端端口: `8501`
- Neo4j: `bolt://localhost:7687`（用户名 neo4j，密码 wyf-agent-2024）
- ChromaDB: `data/chroma/`（持久化）
- JWT 密钥: 环境变量 `JWT_SECRET`（默认 wyf-agent-secret-key-change-in-production）

## 登录账号

- admin / admin123
- user / user123

## 待完成

- 桌面端 Agent 开发（Electron + React）
  - 登录界面 ✅
  - 聊天界面 ✅
  - 图片上传 ✅
  - 系统托盘 ✅
  - 全局快捷键 ✅
  - 构建打包 ⏳（需要 electron-builder 配置）
  - API 地址配置修复 ✅（8080 → 8081）
  - React 19 createRoot API 适配 ✅
  - 资源路径相对路径修复 ✅
  - 隐藏菜单栏 ✅
- 天气工具优化（MCP test_server）✅
  - 模拟数据改为真实 API（wttr.in）✅
  - 中文城市名 URL 编码修复 ✅
- 多 Agent 角色和通信协议（Step 7）⏳（计划已创建）
- 完善测试体系（A/B 测试、基准对比）（Step 9）
