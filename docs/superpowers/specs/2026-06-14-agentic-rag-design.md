# Agentic RAG 设计规格

## 概述

为 WYF Agent 实现基于 LangGraph 的 Agentic RAG 知识检索能力，支持混合格式文档入库、意图路由、口语化查询改写、自反思检索循环。

## 技术选型

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 架构模式 | Agentic RAG | 需要 Agent 自主控制检索策略和迭代 |
| 编排框架 | LangGraph | 状态机天然支持循环/分支；社区成熟 |
| 向量数据库 | ChromaDB | 已有依赖；轻量零配置；适合中等规模 |
| Embedding 模型 | `shibing624/text2vec-base-chinese` | 免费本地；中文效果好 |
| 文档格式 | 混合（Markdown/PDF/TXT） | 覆盖用户所有文档类型 |

## 架构

```
用户提问
    ↓
┌──────────────────────────────────┐
│  LangGraph State Machine         │
│                                  │
│  ① 意图路由                      │  ← 新增：判断走哪条路径
│       ↓                          │
│  ┌─ 闲聊/通用 → 直接 LLM 回答    │
│  ├─ 长文档分析 → 全文塞入上下文   │
│  └─ 知识检索 ↓                   │
│                                  │
│  ② 查询改写（口语→专业术语扩写）  │
│       ↓                          │
│  ③ 向量检索（ChromaDB）           │
│       ↓                          │
│  ④ 重排序                        │
│       ↓                          │
│  ⑤ 自我评估                      │
│       ↓                          │
│  ┌─ 满意 → ⑦ 生成回答            │
│  └─ 不满意 → ⑥ 拆分子问题        │
│         ↓                        │
│     回到 ②（最多 3 轮）           │
└──────────────────────────────────┘
    ↓
带引用的结构化回答
```

### 意图路由规则

| 意图类型 | 判断条件 | 处理方式 |
|---------|---------|---------|
| `chitchat` | 闲聊、打招呼、通用知识问题 | 跳过 RAG，直接 LLM 回答 |
| `doc_analysis` | 用户上传文档且 token 数 < 模型上下文窗口 60% | 全文塞入上下文，不走 RAG |
| `knowledge_qa` | 知识库相关问题（默认） | 走完整 RAG pipeline |

多轮对话处理：当检测到代词引用（"它"、"这个"、"第二种方案"）时，注入最近 N 轮对话历史到 `rewrite_query` 节点，生成自包含的检索查询。

## 模块设计

### 1. Document Loader — `src/rag/loader.py`

职责：加载混合格式文档，统一输出为文本。

- `load_markdown(path)` → 保留标题结构，提取正文
- `load_pdf(path)` → 使用 PyPDF2 或 pdfplumber 提取文本
- `load_text(path)` → 直接读取
- `load_directory(path)` → 递归扫描目录，按扩展名分发

输出格式：
```python
@dataclass
class Document:
    content: str
    metadata: dict[str, Any]  # source, page, title, file_type
```

### 2. Text Splitter — `src/rag/splitter.py`

职责：将长文档切分为语义连贯的小块。

策略：**递归字符切分 + 文档结构感知**（混合方式）
1. 先按文档结构粗切：Markdown 按 `#` 标题分节，PDF 按章节/段落分块
2. 再按 token 数细切：对超长的结构块递归切分至目标大小
3. 切分优先级：`\n\n`（段落）→ `\n`（换行）→ `.`（句子）→ ` `（词）

参数：
- chunk_size = 512 tokens
- chunk_overlap = 64 tokens
- 保留 metadata：来源文件、标题层级、页码

```python
def split_documents(docs: list[Document], chunk_size=512, overlap=64) -> list[Document]
```

### 3. Embedding Service — `src/rag/embeddings.py`

职责：封装 text2vec 模型，提供统一接口。

```python
class EmbeddingService:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese")
    def embed_texts(self, texts: list[str]) -> list[list[float]]
    def embed_query(self, query: str) -> list[float]
```

- 首次调用时懒加载模型
- 支持 batch embed（入库时批量处理）

### 4. Vector Store — `src/rag/vectorstore.py`

职责：ChromaDB 封装层。

```python
class VectorStore:
    def __init__(self, collection_name: str, embedding_service: EmbeddingService)
    def add_documents(self, docs: list[Document]) -> None
    def search(self, query: str, top_k: int = 10) -> list[Document]
    def delete(self, where: dict) -> None
    def get_collection_stats(self) -> dict
```

- 使用 ChromaDB 的 PersistentClient，数据持久化到 `data/chroma/`
- 自动使用 EmbeddingService 生成向量

### 5. Retriever — `src/rag/retriever.py`

职责：检索 + 重排序。

```python
class Retriever:
    def __init__(self, vector_store: VectorStore)
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]
```

重排序策略：Reciprocal Rank Fusion (RRF)
- 对同一查询的多个候选结果按排名加权
- 不依赖额外模型，零成本
- 公式：`score = Σ 1/(k + rank_i)`，k=60

### 6. RAG Graph — `src/rag/graph.py`

职责：LangGraph 状态机，编排完整 Agentic RAG 流程。

状态定义：
```python
class RAGState(TypedDict):
    query: str                      # 原始查询
    intent: str                     # 意图：chitchat / doc_analysis / knowledge_qa
    conversation_history: list[dict] # 最近 N 轮对话历史
    uploaded_doc: str | None        # 用户上传的文档内容（长文档分析时）
    rewritten_query: str            # 改写后的查询
    sub_queries: list[str]          # 拆分的子问题
    retrieved_docs: list[Document]  # 检索结果
    evaluation: str                 # sufficient / needs_refinement / needs_decompose
    answer: str                     # 最终回答
    sources: list[str]              # 引用来源
    iteration: int                  # 当前轮次
```

节点：
1. `route_intent` — LLM 判断意图类型，决定走哪条路径
2. `rewrite_query` — 注入对话历史，将口语化查询改写为专业检索语句
3. `retrieve` — 向量检索 top-10
4. `rerank` — RRF 重排序，输出 top-5
5. `evaluate` — LLM 评估检索结果是否足够回答问题
6. `refine_query` — 改写查询重试（needs_refinement 时）
7. `decompose` — 拆分子问题分别检索（needs_decompose 时）
8. `generate` — 综合所有检索结果生成带引用的回答

边：
```
START → route_intent
route_intent --chitchat--> generate_direct (直接 LLM 回答)
route_intent --doc_analysis--> generate_with_context (全文上下文)
route_intent --knowledge_qa--> rewrite_query → retrieve → rerank → evaluate
evaluate --sufficient--> generate
evaluate --needs_refinement--> refine_query → retrieve
evaluate --needs_decompose--> decompose → rewrite_query
iteration >= 3 --> generate (强制终止)
```

### 7. Knowledge Tool — `src/tools/knowledge.py`

职责：注册到 ToolRegistry 的工具接口，供 ReAct 引擎调用。

```python
class KnowledgeSearchTool(Tool):
    schema: ToolSchema = ToolSchema(
        name="search_knowledge_base",
        description="从知识库中检索相关信息",
        parameters=[
            ToolParameter(name="query", type="string", description="检索查询", required=True),
            ToolParameter(name="top_k", type="integer", description="返回结果数量", required=False),
        ],
    )
    async def execute(self, query: str, top_k: int = 5) -> str
```

## 依赖变更

pyproject.toml 新增：
```toml
"langgraph>=0.2.0"
```

已有依赖（不需要加）：
- `chromadb>=0.6.0`
- `sentence-transformers>=3.4.0`

新增文档解析依赖：
```toml
"pypdf2>=3.0.0"
```

## 数据流

### 文档入库
```
文件路径 → loader.load_*() → splitter.split_documents() → vectorstore.add_documents()
```

### 查询流程
```
用户问题 → route_intent
  ├─ chitchat → 直接 LLM 回答
  ├─ doc_analysis → 全文上下文 + LLM 回答
  └─ knowledge_qa → rewrite_query → retrieve → rerank → evaluate → (refine/decompose) → generate → 带引用回答
```

## 目录结构

```
src/rag/
├── __init__.py
├── loader.py        # 文档加载
├── splitter.py      # 文本分块
├── embeddings.py    # Embedding 服务
├── vectorstore.py   # ChromaDB 封装
├── retriever.py     # 检索+重排序
└── graph.py         # LangGraph 状态机
src/tools/
└── knowledge.py     # 知识检索工具
data/
└── chroma/          # ChromaDB 持久化目录
```

## 测试计划

1. 单元测试：loader、splitter、embeddings、vectorstore、retriever
2. 集成测试：graph 完整流程（mock LLM）
3. 端到端测试：真实文档入库 → 查询 → 验证回答质量

## 范围外

- 前端 UI（后续迭代）
- 文档增量更新/删除管理界面
- 多模态文档（图片、表格）
- 分布式向量存储
