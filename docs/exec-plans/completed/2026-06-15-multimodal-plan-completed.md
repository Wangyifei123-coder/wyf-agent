# 多模态图片支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 升级到 mimo-v2.5 模型，支持图片输入和图片入库

**Architecture:** 模型配置升级 + API 多模态消息格式 + 前端图片组件 + LLM 图片描述入库

**Tech Stack:** FastAPI, Streamlit, LiteLLM, ChromaDB, LangGraph

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `config/.env.example` | 环境变量模板 |
| `config/config.yaml` | 应用配置 |
| `src/gateway/client.py` | LLM 客户端（默认模型名） |
| `src/api.py` | API 端点（ChatRequest） |
| `src/rag/graph.py` | Agentic RAG 状态机 |
| `src/rag/loader.py` | 文档加载器 |
| `src/rag/image_describer.py` | **新增** 图片描述生成器 |
| `frontend.py` | Streamlit 前端 |
| `tests/test_multimodal.py` | **新增** 多模态测试 |

---

## Task 1: 模型配置升级

**Files:**
- Modify: `config/.env.example`
- Modify: `config/config.yaml`
- Modify: `src/gateway/client.py`

- [ ] **Step 1: 更新 .env.example 模型配置**

```bash
# 文件: config/.env.example
# 将 LLM_PRIMARY_MODEL 和 LLM_FALLBACK_MODEL 改为 mimo-v2.5
LLM_PRIMARY_MODEL=anthropic/mimo-v2.5
LLM_FALLBACK_MODEL=anthropic/mimo-v2.5
```

- [ ] **Step 2: 更新 config.yaml 模型配置**

```yaml
# 文件: config/config.yaml
gateway:
  primary_model: "anthropic/mimo-v2.5"
  fallback_model: "anthropic/mimo-v2.5"
```

- [ ] **Step 3: 更新 client.py 默认模型**

```python
# 文件: src/gateway/client.py
@dataclass
class LLMConfig:
    primary_model: str = "anthropic/mimo-v2.5"
    fallback_model: str = "anthropic/mimo-v2.5"
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS（模型配置不影响现有测试）

- [ ] **Step 5: 提交**

```bash
git add config/.env.example config/config.yaml src/gateway/client.py
git commit -m "chore: upgrade model to mimo-v2.5 for multimodal support"
```

---

## Task 2: API 接口支持图片

**Files:**
- Modify: `src/api.py`

- [ ] **Step 1: 更新 ChatRequest 模型**

```python
# 文件: src/api.py
class ChatRequest(BaseModel):
    message: str
    images: list[str] | None = None  # Base64 编码的图片列表
    session_id: str | None = None
```

- [ ] **Step 2: 更新 /chat 端点传递 images**

```python
# 文件: src/api.py
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: str | None = Header(None)) -> ChatResponse:
    # ... 现有代码 ...
    result = await rag_graph.run(
        request.message,
        images=request.images  # 新增
    )
```

- [ ] **Step 3: 更新 /chat/stream 端点传递 images**

```python
# 文件: src/api.py
async def generate() -> Any:
    async for event in rag_graph.run_stream(
        request.message,
        images=request.images  # 新增
    ):
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/api.py
git commit -m "feat: add images field to ChatRequest API"
```

---

## Task 3: Agentic RAG 支持多模态

**Files:**
- Modify: `src/rag/graph.py`

- [ ] **Step 1: RAGState 添加 images 字段**

```python
# 文件: src/rag/graph.py
class RAGState(TypedDict, total=False):
    query: str
    images: list[str] | None  # 新增：Base64 图片列表
    intent: Intent
    conversation_history: list[dict[str, str]]
    uploaded_doc: str | None
    rewritten_query: str
    sub_queries: list[str]
    retrieved_docs: list[Document]
    evaluation: str
    answer: str
    sources: list[str]
    iteration: int
```

- [ ] **Step 2: 意图路由增加图片判断**

```python
# 文件: src/rag/graph.py
async def _route_intent(self, state: RAGState) -> dict[str, Any]:
    query = state["query"]
    uploaded_doc = state.get("uploaded_doc")
    images = state.get("images")

    # 有图片 → 直接走 doc_analysis
    if images:
        logger.info("intent_routed", intent="doc_analysis", reason="has_images")
        return {"intent": Intent.DOC_ANALYSIS}

    # 原有逻辑...
```

- [ ] **Step 3: 构造多模态消息的辅助函数**

```python
# 文件: src/rag/graph.py
def _build_multimodal_messages(
    query: str,
    images: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """构造多模态消息格式"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if images:
        content = [{"type": "text", "text": query}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": query})

    return messages
```

- [ ] **Step 4: 更新 _generate_direct 支持多模态**

```python
# 文件: src/rag/graph.py
async def _generate_direct(self, state: RAGState) -> dict[str, Any]:
    query = state["query"]
    images = state.get("images")
    history = state.get("conversation_history", [])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是一个友好的AI助手,用自然友好的方式回答问题。"},
    ]
    if history:
        messages.extend(history[-6:])

    if images:
        content = [{"type": "text", "text": query}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"}
            })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": query})

    response = await self.llm.chat(messages)

    logger.info("direct_answer_generated")
    return {"answer": response.content, "sources": []}
```

- [ ] **Step 5: 更新 run 和 run_stream 方法签名**

```python
# 文件: src/rag/graph.py
async def run(
    self,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    uploaded_doc: str | None = None,
    images: list[str] | None = None,  # 新增
) -> RAGState:
    initial_state: RAGState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "uploaded_doc": uploaded_doc,
        "images": images,  # 新增
        "iteration": 0,
    }
    # ...

async def run_stream(
    self,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    uploaded_doc: str | None = None,
    images: list[str] | None = None,  # 新增
) -> Any:
    initial_state: RAGState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "uploaded_doc": uploaded_doc,
        "images": images,  # 新增
        "iteration": 0,
    }
    # ...
```

- [ ] **Step 6: 运行测试验证**

Run: `python -m pytest tests/test_rag_graph.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/rag/graph.py
git commit -m "feat: add multimodal support to Agentic RAG graph"
```

---

## Task 4: 前端图片组件

**Files:**
- Modify: `frontend.py`

- [ ] **Step 1: 添加图片上传组件**

```python
# 文件: frontend.py - 在侧边栏添加图片上传
with st.sidebar:
    # ... 现有代码 ...

    st.markdown("---")
    st.subheader("图片上传")
    uploaded_images = st.file_uploader(
        "上传图片（可多选）",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True,
    )
```

- [ ] **Step 2: 在对话区域添加图片预览和粘贴支持**

```python
# 文件: frontend.py - 在对话输入区域添加
st.title("WYF Agent 对话")

# 图片上传区
uploaded_images = st.file_uploader(
    "上传图片",
    type=["png", "jpg", "jpeg", "gif", "webp"],
    accept_multiple_files=True,
    key="chat_images",
)

# 图片预览
images_base64 = []
if uploaded_images:
    cols = st.columns(min(len(uploaded_images), 4))
    for i, img in enumerate(uploaded_images):
        with cols[i % 4]:
            st.image(img, caption=img.name, use_container_width=True)
            import base64
            img_base64 = base64.b64encode(img.getvalue()).decode()
            images_base64.append(img_base64)
```

- [ ] **Step 3: 更新消息发送逻辑传递图片**

```python
# 文件: frontend.py - 更新对话发送
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        if images_base64:
            for img in uploaded_images:
                st.image(img, width=200)

    with st.chat_message("assistant"):
        # ... 现有代码 ...
        with client.stream(
            "POST",
            f"{API_BASE}/chat/stream",
            json={"message": prompt, "images": images_base64 or None},
            headers=headers,
        ) as response:
            # ...
```

- [ ] **Step 4: 运行前端验证**

Run: `python -m streamlit run frontend.py --server.port 8501`
Expected: 前端正常显示图片上传组件

- [ ] **Step 5: 提交**

```bash
git add frontend.py
git commit -m "feat: add image upload and preview to frontend"
```

---

## Task 5: 图片入库（LLM 描述）

**Files:**
- Create: `src/rag/image_describer.py`
- Modify: `src/rag/loader.py`
- Modify: `src/api.py`

- [ ] **Step 1: 创建 image_describer.py**

```python
# 文件: src/rag/image_describer.py
"""图片描述生成器 — 用 LLM 为图片生成文本描述"""

from __future__ import annotations

import base64
from pathlib import Path

import structlog

from src.gateway.client import LLMClient

logger = structlog.get_logger(__name__)

DESCRIBE_PROMPT = """请详细描述这张图片的内容，包括：
1. 图片类型（照片、图表、截图等）
2. 主要内容和对象
3. 文字信息（如果有）
4. 其他重要细节

描述："""


async def describe_image(llm: LLMClient, image_path: str) -> str:
    """用 LLM 生成图片描述"""
    image_data = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_data).decode()

    ext = Path(image_path).suffix.lower().replace(".", "")
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
            ],
        }
    ]

    response = await llm.chat(messages)
    description = response.content.strip()

    logger.info("image_described", path=image_path, description_len=len(description))
    return description
```

- [ ] **Step 2: 更新 loader.py 支持图片**

```python
# 文件: src/rag/loader.py
# 在 EXTENSION_MAP 中添加图片格式
EXTENSION_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
}
```

- [ ] **Step 3: 更新 API 端点支持图片入库**

```python
# 文件: src/api.py
@app.post("/knowledge/ingest", response_model=IngestResponse)
async def ingest_knowledge(request: IngestRequest) -> IngestResponse:
    assert vector_store and hybrid_retriever and llm_client
    from .rag.image_describer import describe_image
    from .rag.splitter import split_documents

    docs = load_directory(request.path)

    # 处理图片文件
    image_docs = []
    text_docs = []
    for doc in docs:
        if doc.metadata.get("file_type") == "image":
            description = await describe_image(llm_client, doc.metadata["source"])
            doc.content = description
            doc.metadata["file_type"] = "image_description"
            image_docs.append(doc)
        else:
            text_docs.append(doc)

    all_docs = text_docs + image_docs
    chunks = split_documents(all_docs)

    # ... 现有入库逻辑 ...
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/rag/image_describer.py src/rag/loader.py src/api.py
git commit -m "feat: add image ingestion with LLM description"
```

---

## Task 6: 端到端测试

**Files:**
- Create: `tests/test_multimodal.py`

- [ ] **Step 1: 创建多模态测试文件**

```python
# 文件: tests/test_multimodal.py
"""多模态功能测试"""

import pytest
from src.rag.graph import RAGState, Intent


def test_rag_state_has_images_field():
    """RAGState 应该支持 images 字段"""
    state: RAGState = {
        "query": "test",
        "images": ["base64data"],
    }
    assert state.get("images") == ["base64data"]


def test_rag_state_images_optional():
    """images 字段应该是可选的"""
    state: RAGState = {"query": "test"}
    assert state.get("images") is None
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/test_multimodal.py -v`
Expected: PASS

- [ ] **Step 3: 运行全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（57+ 个测试）

- [ ] **Step 4: 运行代码检查**

Run: `python -m ruff check src/`
Run: `python -m mypy src/`
Expected: 无错误

- [ ] **Step 5: 提交**

```bash
git add tests/test_multimodal.py
git commit -m "test: add multimodal feature tests"
```

---

## Task 7: 更新文档

**Files:**
- Modify: `session-handoff.md`
- Modify: `docs/QUALITY_SCORE.md`

- [ ] **Step 1: 更新 session-handoff.md**

```markdown
## 当前已验证

- ... 现有内容 ...
- **多模态图片支持**（2026-06-15）
  - 模型升级到 mimo-v2.5 ✅
  - API 支持 images 字段 ✅
  - 前端图片上传/预览 ✅
  - 图片入库（LLM 描述）✅
```

- [ ] **Step 2: 更新质量评分**

```markdown
| Chat (基础对话) | 已验证 | 高 | 8 测试通过 | B → B+ |
| Tools (工具调用) | 未验证 | 高 | 无测试 | D |
| Memory (记忆) | 未验证 | 高 | 无测试 | D |
```

- [ ] **Step 3: 提交**

```bash
git add session-handoff.md docs/QUALITY_SCORE.md
git commit -m "docs: update progress with multimodal support"
```

---

## 验收检查清单

- [ ] 模型配置更新后，对话正常工作
- [ ] 前端可以上传图片，显示预览
- [ ] 发送带图片的问题，LLM 能正确分析图片内容
- [ ] 图片入库后，后续问题能检索到图片相关内容
- [ ] 现有 57 个测试继续通过
- [ ] ruff check 无错误
- [ ] mypy 无错误
- [ ] 文档已更新
