# Session Handoff

## 当前已验证

- 项目骨架已创建，7 层架构完整
- pyproject.toml 配置完成，依赖安装成功
- 57/57 测试通过（基础 + RAG 全组件）
- LLM 调用验证通过（mimo-v2.5-pro via Anthropic 兼容接口）
- FastAPI 服务启动正常
- `/health`、`/chat`、`/knowledge/ingest`、`/knowledge/stats` 端点验证通过
- **ruff check 全部通过**
- **mypy 全部通过**
- Agentic RAG 全流程验证通过（意图路由、查询改写、自反思循环）
- 三路混合检索验证通过（向量 + BM25 + 知识图谱）
- Neo4j Docker 部署并验证连接
- 评估数据集 20 组，评估得分 0.92
- Streamlit 前端对话界面完成

## 本轮改动

- 实现 Agentic RAG 全流程（LangGraph 状态机）
- 实现三路混合检索（向量 + BM25 + Neo4j 知识图谱，RRF 融合）
- 创建评估数据集（20 组测试用例）+ 评估脚本
- 修复 Tool Call 泄露（/chat 直接走 RAGGraph）
- 优化查询改写（代词消解）和问题分解（多文档检索）
- 创建 Streamlit 前端对话界面
- 创建 Docker 部署配置（docker-compose）

## 新增文件

```
src/rag/
├── __init__.py
├── loader.py           # 文档加载（Markdown/PDF/TXT）
├── splitter.py         # 结构感知文本分块
├── embeddings.py       # text2vec 中文 Embedding
├── vectorstore.py      # ChromaDB 封装
├── retriever.py        # RRF 重排序检索
├── bm25_retriever.py   # BM25 关键词检索
├── kg_retriever.py     # Neo4j 知识图谱检索
├── hybrid_retriever.py # 三路 RRF 融合
└── graph.py            # LangGraph Agentic RAG 状态机
src/tools/
└── knowledge.py        # 知识检索工具
evals/
├── test_dataset.json   # 20 组评估用例
├── run_eval.py         # 评估脚本
└── eval_results.json   # 评估结果
frontend.py             # Streamlit 前端
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

## 配置

- API Key: `config/.env`（已 gitignore）
- 模型: `anthropic/mimo-v2.5-pro`
- 端点: `https://token-plan-cn.xiaomimimo.com/anthropic`
- 后端端口: `8080`
- 前端端口: `8501`
- Neo4j: `bolt://localhost:7687`（用户名 neo4j，密码 wyf-agent-2024）
- ChromaDB: `data/chroma/`（持久化）

## 待完成

- 更多文档入库测试
- 评估数据集扩充（20 → 50+）
- 增量更新支持
- 评估优化（多文档检索、代词消解持续改进）
