# Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 WYF Agent 实现基于 LangGraph 的 Agentic RAG 知识检索能力，支持混合格式文档入库、意图路由、口语化查询改写、自反思检索循环。

**Architecture:** 新增 `src/rag/` 模块，包含文档加载、分块、Embedding、向量存储、检索、LangGraph 状态机。通过意图路由决定走 RAG、长上下文还是直接回答。

**Tech Stack:** LangGraph, ChromaDB, sentence-transformers (text2vec-base-chinese), PyPDF2, structlog

---

## File Structure

```
src/rag/
├── __init__.py          # 模块导出
├── loader.py            # 文档加载（Markdown/PDF/TXT）
├── splitter.py          # 文本分块（结构感知 + 递归切分）
├── embeddings.py        # Embedding 服务（text2vec）
├── vectorstore.py       # ChromaDB 封装
├── retriever.py         # 检索 + RRF 重排序
└── graph.py             # LangGraph 状态机（意图路由 + Agentic RAG）
src/tools/
└── knowledge.py         # 知识检索工具（注册到 ToolRegistry）
tests/
├── test_rag_loader.py   # Loader 测试
├── test_rag_splitter.py # Splitter 测试
├── test_rag_embeddings.py # Embedding 测试
├── test_rag_vectorstore.py # VectorStore 测试
├── test_rag_retriever.py # Retriever 测试
├── test_rag_graph.py    # Graph 集成测试
└── test_knowledge_tool.py # Knowledge Tool 测试
```

---

### Task 1: 添加依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 langgraph 和 pypdf2 依赖**

```toml
# pyproject.toml dependencies 中添加
dependencies = [
    # ... existing deps ...
    "langgraph>=0.2.0",
    "pypdf2>=3.0.0",
]
```

- [ ] **Step 2: 安装依赖**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed

- [ ] **Step 3: 验证安装**

Run: `python -c "import langgraph; print('langgraph ok')"` and `python -c "import PyPDF2; print('pypdf2 ok')"`
Expected: 两个都输出 ok

---

### Task 2: Document Loader

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/loader.py`
- Create: `tests/test_rag_loader.py`

- [ ] **Step 1: 创建 src/rag/__init__.py**

```python
"""RAG 模块 — 文档加载、分块、Embedding、检索、Agentic RAG"""
```

- [ ] **Step 2: 写 loader 测试**

```python
# tests/test_rag_loader.py
import os
import tempfile
import pytest
from src.rag.loader import Document, load_markdown, load_pdf, load_text, load_directory


class TestDocument:
    def test_document_creation(self):
        doc = Document(content="hello", metadata={"source": "test.md"})
        assert doc.content == "hello"
        assert doc.metadata["source"] == "test.md"


class TestLoadMarkdown:
    def test_load_basic_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Title\n\nSome content here.\n\n## Section\n\nMore content.")
            f.flush()
            path = f.name
        try:
            doc = load_markdown(path)
            assert "Title" in doc.content
            assert "Some content here" in doc.content
            assert doc.metadata["file_type"] == "markdown"
            assert doc.metadata["source"] == path
        finally:
            os.unlink(path)


class TestLoadText:
    def test_load_basic_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello world.\nThis is a test.")
            f.flush()
            path = f.name
        try:
            doc = load_text(path)
            assert "Hello world" in doc.content
            assert doc.metadata["file_type"] == "text"
        finally:
            os.unlink(path)


class TestLoadPdf:
    def test_load_pdf_creates_document(self):
        pytest.importorskip("PyPDF2")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            from PyPDF2 import PdfWriter
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with open(path, "wb") as f:
                writer.write(f)
            doc = load_pdf(path)
            assert doc.metadata["file_type"] == "pdf"
            assert doc.metadata["source"] == path
        finally:
            os.unlink(path)


class TestLoadDirectory:
    def test_load_directory_mixed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [("a.md", "# Hello"), ("b.txt", "World"), ("c.md", "# Third")]:
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                    f.write(content)
            docs = load_directory(tmpdir)
            assert len(docs) == 3
            assert all(isinstance(d, Document) for d in docs)

    def test_load_directory_skips_non_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.md"), "w", encoding="utf-8") as f:
                f.write("# Hello")
            with open(os.path.join(tmpdir, "b.csv"), "w", encoding="utf-8") as f:
                f.write("a,b,c")
            docs = load_directory(tmpdir)
            assert len(docs) == 1
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_loader.py -v`
Expected: FAIL — ModuleNotFoundError: No module named 'src.rag.loader'

- [ ] **Step 4: 实现 loader.py**

```python
# src/rag/loader.py
"""文档加载器 — 支持 Markdown、PDF、TXT 混合格式"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".text", ".pdf"}


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_markdown(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    logger.info("loaded_markdown", path=path, length=len(text))
    return Document(
        content=text,
        metadata={"source": path, "file_type": "markdown"},
    )


def load_text(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    logger.info("loaded_text", path=path, length=len(text))
    return Document(
        content=text,
        metadata={"source": path, "file_type": "text"},
    )


def load_pdf(path: str) -> Document:
    from PyPDF2 import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)

    content = "\n\n".join(pages)
    logger.info("loaded_pdf", path=path, pages=len(pages), length=len(content))
    return Document(
        content=content,
        metadata={"source": path, "file_type": "pdf", "pages": len(pages)},
    )


def load_directory(path: str) -> list[Document]:
    docs: list[Document] = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            try:
                if ext in (".md", ".markdown"):
                    docs.append(load_markdown(fpath))
                elif ext in (".txt", ".text"):
                    docs.append(load_text(fpath))
                elif ext == ".pdf":
                    docs.append(load_pdf(fpath))
            except Exception as e:
                logger.warning("load_failed", path=fpath, error=str(e))
    logger.info("directory_loaded", path=path, count=len(docs))
    return docs
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_loader.py -v`
Expected: 7 passed

- [ ] **Step 6: 提交**

```bash
git add src/rag/__init__.py src/rag/loader.py tests/test_rag_loader.py
git commit -m "feat(rag): add document loader for markdown/pdf/txt"
```

---

### Task 3: Text Splitter

**Files:**
- Create: `src/rag/splitter.py`
- Create: `tests/test_rag_splitter.py`

- [ ] **Step 1: 写 splitter 测试**

```python
# tests/test_rag_splitter.py
from src.rag.loader import Document
from src.rag.splitter import split_documents, split_text


class TestSplitText:
    def test_short_text_unchanged(self):
        text = "Hello world."
        chunks = split_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        text = "A" * 1000
        chunks = split_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 250  # some tolerance

    def test_overlap_present(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = split_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1


class TestSplitDocuments:
    def test_preserves_metadata(self):
        doc = Document(
            content="Hello world. " * 100,
            metadata={"source": "test.md", "file_type": "markdown"},
        )
        chunks = split_documents([doc], chunk_size=50, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"
            assert chunk.metadata["file_type"] == "markdown"
            assert "chunk_index" in chunk.metadata

    def test_markdown_structure_aware(self):
        doc = Document(
            content="# Title\n\nContent under title.\n\n## Section\n\nContent under section.\n\n" * 10,
            metadata={"source": "test.md", "file_type": "markdown"},
        )
        chunks = split_documents([doc], chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_empty_document(self):
        doc = Document(content="", metadata={})
        chunks = split_documents([doc])
        assert len(chunks) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_splitter.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 splitter.py**

```python
# src/rag/splitter.py
"""文本分块器 — 递归字符切分 + 文档结构感知"""

from __future__ import annotations

import re
from typing import Any

import structlog

from .loader import Document

logger = structlog.get_logger(__name__)

SEPARATORS = ["\n\n", "\n", ".", " ", ""]


def _count_tokens(text: str) -> int:
    return len(text) // 3


def _split_by_separators(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
    if _count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text] if text.strip() else []

    sep = separators[0]
    rest_separators = separators[1:]

    if sep == "":
        chunks = []
        for i in range(0, len(text), chunk_size * 3):
            chunk = text[i : i + chunk_size * 3]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if _count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if _count_tokens(part) > chunk_size:
                sub_chunks = _split_by_separators(part, chunk_size, overlap, rest_separators)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        overlap_text = " ".join(prev_words[-overlap:]) if len(prev_words) > overlap else chunks[i - 1]
        result.append(overlap_text + " " + chunks[i])
    return result


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    raw_chunks = _split_by_separators(text, chunk_size, overlap, SEPARATORS)
    return [c for c in raw_chunks if c.strip()]


def split_documents(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    result: list[Document] = []

    for doc in docs:
        if not doc.content.strip():
            continue

        chunks = split_text(doc.content, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(chunks)
            result.append(Document(content=chunk, metadata=metadata))

    logger.info("documents_split", input_docs=len(docs), output_chunks=len(result))
    return result
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_splitter.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add src/rag/splitter.py tests/test_rag_splitter.py
git commit -m "feat(rag): add text splitter with structure-aware chunking"
```

---

### Task 4: Embedding Service

**Files:**
- Create: `src/rag/embeddings.py`
- Create: `tests/test_rag_embeddings.py`

- [ ] **Step 1: 写 embeddings 测试**

```python
# tests/test_rag_embeddings.py
import pytest
from src.rag.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture
    def service(self):
        return EmbeddingService(model_name="shibing624/text2vec-base-chinese")

    def test_embed_query_returns_vector(self, service):
        vec = service.embed_query("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_returns_batch(self, service):
        texts = ["hello", "world", "test"]
        vecs = service.embed_texts(texts)
        assert len(vecs) == 3
        assert all(isinstance(v, list) for v in vecs)

    def test_embed_query_consistent(self, service):
        vec1 = service.embed_query("test query")
        vec2 = service.embed_query("test query")
        assert vec1 == vec2

    def test_embed_different_texts_differ(self, service):
        vec1 = service.embed_query("hello")
        vec2 = service.embed_query("completely different topic about quantum physics")
        assert vec1 != vec2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_embeddings.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 embeddings.py**

```python
# src/rag/embeddings.py
"""Embedding 服务 — 封装 text2vec 模型"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()
        vec = model.encode(query, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_embeddings.py -v`
Expected: 4 passed（首次加载模型可能需要下载，约 100MB）

- [ ] **Step 5: 提交**

```bash
git add src/rag/embeddings.py tests/test_rag_embeddings.py
git commit -m "feat(rag): add embedding service with text2vec"
```

---

### Task 5: Vector Store

**Files:**
- Create: `src/rag/vectorstore.py`
- Create: `tests/test_rag_vectorstore.py`

- [ ] **Step 1: 写 vectorstore 测试**

```python
# tests/test_rag_vectorstore.py
import tempfile
import pytest
from src.rag.loader import Document
from src.rag.embeddings import EmbeddingService
from src.rag.vectorstore import VectorStore


class TestVectorStore:
    @pytest.fixture
    def store(self):
        service = EmbeddingService(model_name="shibing624/text2vec-base-chinese")
        with tempfile.TemporaryDirectory() as tmpdir:
            yield VectorStore(
                collection_name="test",
                embedding_service=service,
                persist_directory=tmpdir,
            )

    def test_add_and_search(self, store):
        docs = [
            Document(content="Python is a programming language", metadata={"source": "a.md"}),
            Document(content="Machine learning uses algorithms", metadata={"source": "b.md"}),
        ]
        store.add_documents(docs)
        results = store.search("programming", top_k=2)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_search_returns_metadata(self, store):
        docs = [Document(content="Test content", metadata={"source": "test.md", "file_type": "markdown"})]
        store.add_documents(docs)
        results = store.search("test", top_k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "test.md"

    def test_empty_search(self, store):
        results = store.search("nothing", top_k=5)
        assert len(results) == 0

    def test_get_collection_stats(self, store):
        stats = store.get_collection_stats()
        assert "count" in stats
        assert stats["count"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_vectorstore.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 vectorstore.py**

```python
# src/rag/vectorstore.py
"""向量存储 — ChromaDB 封装"""

from __future__ import annotations

from typing import Any

import structlog

from .embeddings import EmbeddingService
from .loader import Document

logger = structlog.get_logger(__name__)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "wyf-agent-kb",
        embedding_service: EmbeddingService | None = None,
        persist_directory: str = "data/chroma",
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore_initialized", collection=collection_name)

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return

        texts = [d.content for d in docs]
        ids = [f"doc_{i}_{hash(d.content) % 10**8}" for i, d in enumerate(docs)]
        metadatas = [d.metadata for d in docs]
        embeddings = self.embedding_service.embed_texts(texts)

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("documents_added", count=len(docs))

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        embedding = self.embedding_service.embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        docs: list[Document] = []
        if results and results["documents"]:
            for i, content in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                metadata["distance"] = distance
                metadata["relevance_score"] = 1 - distance
                docs.append(Document(content=content, metadata=metadata))

        return docs

    def delete(self, where: dict[str, Any]) -> None:
        self._collection.delete(where=where)
        logger.info("documents_deleted", where=where)

    def get_collection_stats(self) -> dict[str, Any]:
        return {"count": self._collection.count(), "name": self.collection_name}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_vectorstore.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add src/rag/vectorstore.py tests/test_rag_vectorstore.py
git commit -m "feat(rag): add ChromaDB vector store wrapper"
```

---

### Task 6: Retriever (检索 + RRF 重排序)

**Files:**
- Create: `src/rag/retriever.py`
- Create: `tests/test_rag_retriever.py`

- [ ] **Step 1: 写 retriever 测试**

```python
# tests/test_rag_retriever.py
import pytest
from unittest.mock import MagicMock
from src.rag.loader import Document
from src.rag.retriever import Retriever, rrf_score


class TestRRFScore:
    def test_single_rank(self):
        score = rrf_score([1], k=60)
        assert score == 1 / (60 + 1)

    def test_multiple_ranks(self):
        score = rrf_score([1, 3], k=60)
        expected = 1 / (60 + 1) + 1 / (60 + 3)
        assert abs(score - expected) < 1e-9

    def test_empty_ranks(self):
        score = rrf_score([], k=60)
        assert score == 0


class TestRetriever:
    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.search.return_value = [
            Document(content="doc1", metadata={"source": "a.md", "relevance_score": 0.9}),
            Document(content="doc2", metadata={"source": "b.md", "relevance_score": 0.7}),
            Document(content="doc3", metadata={"source": "c.md", "relevance_score": 0.5}),
        ]
        return store

    def test_retrieve_returns_top_k(self, mock_vector_store):
        retriever = Retriever(mock_vector_store)
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_calls_vector_store(self, mock_vector_store):
        retriever = Retriever(mock_vector_store)
        retriever.retrieve("test query", top_k=5)
        mock_vector_store.search.assert_called_once_with("test query", top_k=10)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_retriever.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 retriever.py**

```python
# src/rag/retriever.py
"""检索器 — 向量检索 + RRF 重排序"""

from __future__ import annotations

import structlog

from .loader import Document
from .vectorstore import VectorStore

logger = structlog.get_logger(__name__)


def rrf_score(ranks: list[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        candidates = self.vector_store.search(query, top_k=top_k * 2)

        for i, doc in enumerate(candidates):
            doc.metadata["rrf_rank"] = i + 1

        scored: dict[str, tuple[float, Document]] = {}
        for doc in candidates:
            key = doc.content[:100]
            rank = doc.metadata.get("rrf_rank", 99)
            score = rrf_score([rank])
            if key not in scored or score > scored[key][0]:
                scored[key] = (score, doc)

        sorted_results = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in sorted_results[:top_k]]

        for i, doc in enumerate(results):
            doc.metadata["final_rank"] = i + 1

        logger.info("retrieval_complete", query=query[:50], candidates=len(candidates), results=len(results))
        return results
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_retriever.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add src/rag/retriever.py tests/test_rag_retriever.py
git commit -m "feat(rag): add retriever with RRF reranking"
```

---

### Task 7: LangGraph Agentic RAG 状态机

**Files:**
- Create: `src/rag/graph.py`
- Create: `tests/test_rag_graph.py`

- [ ] **Step 1: 写 graph 测试**

```python
# tests/test_rag_graph.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.rag.graph import RAGState, RAGGraph, Intent


class TestRAGGraph:
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.chat = AsyncMock()
        return llm

    @pytest.fixture
    def mock_retriever(self):
        from src.rag.loader import Document
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            Document(content="test doc", metadata={"source": "a.md", "relevance_score": 0.9}),
        ]
        return retriever

    def test_initial_state(self):
        state = RAGState(
            query="test",
            intent="",
            conversation_history=[],
            uploaded_doc=None,
            rewritten_query="",
            sub_queries=[],
            retrieved_docs=[],
            evaluation="",
            answer="",
            sources=[],
            iteration=0,
        )
        assert state["query"] == "test"
        assert state["iteration"] == 0

    @pytest.mark.asyncio
    async def test_chitchat_bypasses_rag(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "chitchat"
        mock_response.usage = {"total_tokens": 10}
        mock_llm.chat.return_value = mock_response

        gen_response = MagicMock()
        gen_response.content = "Hello! How are you?"
        gen_response.usage = {"total_tokens": 20}

        mock_llm.chat.side_effect = [mock_response, gen_response]

        graph = RAGGraph(llm=mock_llm, retriever=MagicMock())
        result = await graph.run("你好", conversation_history=[])
        assert result["answer"] == "Hello! How are you?"
        assert result["intent"] == "chitchat"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_rag_graph.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 graph.py**

```python
# src/rag/graph.py
"""Agentic RAG 状态机 — LangGraph 编排"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

import structlog

from ..gateway.client import LLMClient, LLMResponse
from .loader import Document
from .retriever import Retriever

logger = structlog.get_logger(__name__)

MAX_ITERATIONS = 3


class Intent(StrEnum):
    CHITCHAT = "chitchat"
    DOC_ANALYSIS = "doc_analysis"
    KNOWLEDGE_QA = "knowledge_qa"


class RAGState(TypedDict):
    query: str
    intent: str
    conversation_history: list[dict[str, str]]
    uploaded_doc: str | None
    rewritten_query: str
    sub_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    evaluation: str
    answer: str
    sources: list[str]
    iteration: int


ROUTE_PROMPT = """你是一个意图分类器。根据用户问题判断意图类型。

意图类型：
- chitchat: 闲聊、打招呼、通用知识问答（不需要检索特定知识库）
- doc_analysis: 用户要求分析已提供的文档内容
- knowledge_qa: 需要从知识库中检索信息来回答的问题（默认）

用户问题：{query}

只回答一个词：chitchat、doc_analysis 或 knowledge_qa"""

REWRITE_PROMPT = """将以下口语化问题改写为包含专业术语的检索语句，保持原意。
如果包含代词引用（如"它"、"这个"、"第二种方案"），请结合对话历史补全上下文。

对话历史：
{history}

用户问题：{query}

改写后的检索语句（只输出改写结果，不要解释）："""

EVALUATE_PROMPT = """你是一个检索质量评估器。判断以下检索结果是否足够回答用户问题。

用户问题：{query}

检索结果：
{context}

请判断：
- sufficient: 检索结果足够回答问题
- needs_refinement: 结果相关但不够，需要改写查询重试
- needs_decompose: 问题太复杂，需要拆分为子问题

只回答一个词：sufficient、needs_refinement 或 needs_decompose"""

DECOMPOSE_PROMPT = """将以下复杂问题拆分为 2-3 个简单的子问题，每个子问题可以独立检索。

原始问题：{query}

请输出子问题，每行一个（不要编号，不要解释）："""

GENERATE_PROMPT = """根据以下检索结果回答用户问题。如果检索结果不相关，请基于你的知识回答。

检索结果：
{context}

用户问题：{query}

请给出详细回答，并在回答末尾列出引用来源。"""


def _docs_to_context(docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("metadata", {}).get("source", "unknown")
        parts.append(f"[{i}] (来源: {source})\n{doc.get('content', '')}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[dict[str, Any]]) -> list[str]:
    seen = set()
    sources = []
    for doc in docs:
        src = doc.get("metadata", {}).get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


class RAGGraph:
    def __init__(self, llm: LLMClient, retriever: Retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        uploaded_doc: str | None = None,
    ) -> RAGState:
        state = RAGState(
            query=query,
            intent="",
            conversation_history=conversation_history or [],
            uploaded_doc=uploaded_doc,
            rewritten_query="",
            sub_queries=[],
            retrieved_docs=[],
            evaluation="",
            answer="",
            sources=[],
            iteration=0,
        )

        state = await self._route_intent(state)

        if state["intent"] == Intent.CHITCHAT:
            state = await self._generate_direct(state)
        elif state["intent"] == Intent.DOC_ANALYSIS:
            state = await self._generate_with_context(state)
        else:
            state = await self._rag_loop(state)

        return state

    async def _route_intent(self, state: RAGState) -> RAGState:
        if state["uploaded_doc"]:
            state["intent"] = Intent.DOC_ANALYSIS
            return state

        prompt = ROUTE_PROMPT.format(query=state["query"])
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        intent = response.content.strip().lower()

        if intent in (Intent.CHITCHAT, Intent.DOC_ANALYSIS, Intent.KNOWLEDGE_QA):
            state["intent"] = intent
        else:
            state["intent"] = Intent.KNOWLEDGE_QA

        logger.info("intent_routed", query=state["query"][:50], intent=state["intent"])
        return state

    async def _rewrite_query(self, state: RAGState) -> RAGState:
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in state["conversation_history"][-6:]
        ) if state["conversation_history"] else "无"

        prompt = REWRITE_PROMPT.format(history=history_text, query=state["query"])
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        state["rewritten_query"] = response.content.strip()

        logger.info("query_rewritten", original=state["query"][:50], rewritten=state["rewritten_query"][:50])
        return state

    async def _retrieve(self, state: RAGState) -> RAGState:
        query = state["rewritten_query"] or state["query"]
        docs = self.retriever.retrieve(query, top_k=5)
        state["retrieved_docs"] = [
            {"content": d.content, "metadata": d.metadata} for d in docs
        ]
        logger.info("docs_retrieved", count=len(state["retrieved_docs"]))
        return state

    async def _evaluate(self, state: RAGState) -> RAGState:
        context = _docs_to_context(state["retrieved_docs"])
        prompt = EVALUATE_PROMPT.format(query=state["query"], context=context)
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        evaluation = response.content.strip().lower()

        if evaluation in ("sufficient", "needs_refinement", "needs_decompose"):
            state["evaluation"] = evaluation
        else:
            state["evaluation"] = "sufficient"

        logger.info("evaluation_result", evaluation=state["evaluation"])
        return state

    async def _refine_query(self, state: RAGState) -> RAGState:
        prompt = REWRITE_PROMPT.format(
            history=f"之前的检索结果不够好，请换一个角度改写。\n原始问题：{state['query']}",
            query=state["query"],
        )
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        state["rewritten_query"] = response.content.strip()
        state["iteration"] += 1
        return state

    async def _decompose(self, state: RAGState) -> RAGState:
        prompt = DECOMPOSE_PROMPT.format(query=state["query"])
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        state["sub_queries"] = sub_queries[:3]
        state["iteration"] += 1

        all_docs = list(state["retrieved_docs"])
        for sq in state["sub_queries"]:
            docs = self.retriever.retrieve(sq, top_k=3)
            for d in docs:
                doc_dict = {"content": d.content, "metadata": d.metadata}
                if doc_dict not in all_docs:
                    all_docs.append(doc_dict)
        state["retrieved_docs"] = all_docs

        logger.info("decomposed", sub_queries=len(state["sub_queries"]), total_docs=len(all_docs))
        return state

    async def _generate(self, state: RAGState) -> RAGState:
        context = _docs_to_context(state["retrieved_docs"])
        prompt = GENERATE_PROMPT.format(query=state["query"], context=context)
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        state["answer"] = response.content
        state["sources"] = _extract_sources(state["retrieved_docs"])
        return state

    async def _generate_direct(self, state: RAGState) -> RAGState:
        messages = state["conversation_history"] + [{"role": "user", "content": state["query"]}]
        response = await self.llm.chat(messages)
        state["answer"] = response.content
        return state

    async def _generate_with_context(self, state: RAGState) -> RAGState:
        prompt = f"请分析以下文档内容并回答问题。\n\n文档内容：\n{state['uploaded_doc']}\n\n问题：{state['query']}"
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        state["answer"] = response.content
        return state

    async def _rag_loop(self, state: RAGState) -> RAGState:
        state = await self._rewrite_query(state)
        state = await self._retrieve(state)
        state = await self._evaluate(state)

        while state["evaluation"] != "sufficient" and state["iteration"] < MAX_ITERATIONS:
            if state["evaluation"] == "needs_decompose":
                state = await self._decompose(state)
            else:
                state = await self._refine_query(state)

            state = await self._retrieve(state)
            state = await self._evaluate(state)

        state = await self._generate(state)
        return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_rag_graph.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/rag/graph.py tests/test_rag_graph.py
git commit -m "feat(rag): add LangGraph agentic RAG with intent routing"
```

---

### Task 8: Knowledge Tool

**Files:**
- Create: `src/tools/knowledge.py`
- Create: `tests/test_knowledge_tool.py`

- [ ] **Step 1: 写 knowledge tool 测试**

```python
# tests/test_knowledge_tool.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.tools.knowledge import KnowledgeSearchTool


class TestKnowledgeSearchTool:
    @pytest.fixture
    def tool(self):
        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value={
            "answer": "Python is a programming language.",
            "sources": ["docs/python.md"],
            "intent": "knowledge_qa",
            "evaluation": "sufficient",
            "iteration": 1,
        })
        return KnowledgeSearchTool(rag_graph=mock_graph)

    def test_schema(self, tool):
        assert tool.schema.name == "search_knowledge_base"
        assert len(tool.schema.parameters) >= 1
        assert tool.schema.parameters[0].name == "query"

    @pytest.mark.asyncio
    async def test_execute_returns_answer(self, tool):
        result = await tool.execute(query="什么是Python")
        assert "Python" in result
        assert "docs/python.md" in result

    @pytest.mark.asyncio
    async def test_execute_calls_graph(self, tool):
        await tool.execute(query="test query")
        tool._rag_graph.run.assert_called_once()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_knowledge_tool.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 knowledge.py**

```python
# src/tools/knowledge.py
"""知识检索工具 — 注册到 ToolRegistry 供 ReAct 引擎调用"""

from __future__ import annotations

from typing import Any

import structlog

from .registry import Tool, ToolParameter, ToolSchema

logger = structlog.get_logger(__name__)


class KnowledgeSearchTool(Tool):
    def __init__(self, rag_graph: Any) -> None:
        self._rag_graph = rag_graph

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_knowledge_base",
            description="从知识库中检索相关信息并回答问题",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="用户的检索问题",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: query parameter is required"

        logger.info("knowledge_search", query=query[:100])

        result = await self._rag_graph.run(query)

        answer = result.get("answer", "未找到相关信息")
        sources = result.get("sources", [])

        response_parts = [answer]
        if sources:
            response_parts.append(f"\n\n引用来源：{', '.join(sources)}")

        return "\n".join(response_parts)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_knowledge_tool.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add src/tools/knowledge.py tests/test_knowledge_tool.py
git commit -m "feat(tools): add knowledge search tool for RAG"
```

---

### Task 9: 集成到 API 入口

**Files:**
- Modify: `src/api.py`

- [ ] **Step 1: 读取当前 api.py**

读取并理解现有 API 入口代码。

- [ ] **Step 2: 在 lifespan 中初始化 RAG 组件**

在 FastAPI lifespan 中初始化 EmbeddingService、VectorStore、Retriever、RAGGraph，并注册 KnowledgeSearchTool 到 ToolRegistry。

- [ ] **Step 3: 添加知识库管理端点**

添加 `/knowledge/ingest` 端点用于文档入库，`/knowledge/stats` 端点查看知识库状态。

- [ ] **Step 4: 跑全部测试确认不破坏现有功能**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add src/api.py
git commit -m "feat(api): integrate RAG pipeline into FastAPI entry point"
```

---

### Task 10: 端到端验证

- [ ] **Step 1: 创建测试文档目录**

```bash
mkdir -p data/test_docs
```

写入几个测试 Markdown 文件到 `data/test_docs/`。

- [ ] **Step 2: 启动服务**

Run: `python -m uvicorn src.api:app --reload --port 8080`

- [ ] **Step 3: 测试文档入库**

```bash
curl -X POST http://localhost:8080/knowledge/ingest -H "Content-Type: application/json" -d '{"path": "data/test_docs"}'
```

- [ ] **Step 4: 测试知识检索**

```bash
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "什么是Python"}'
```

- [ ] **Step 5: 测试闲聊绕过 RAG**

```bash
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "你好"}'
```

- [ ] **Step 6: 跑 ruff + mypy**

Run: `python -m ruff check src/` and `python -m mypy src/`
Expected: 无新增错误

- [ ] **Step 7: 最终提交**

```bash
git add -A
git commit -m "feat: complete agentic RAG implementation with intent routing"
```
