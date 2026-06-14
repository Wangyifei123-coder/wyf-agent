"""BM25 检索器测试"""

from src.rag.bm25_retriever import BM25Retriever
from src.rag.loader import Document


class TestBM25Retriever:
    def test_search_returns_relevant_docs(self):
        retriever = BM25Retriever()
        docs = [
            Document(content="Python是一种编程语言", metadata={"source": "a.md"}),
            Document(content="RAG是检索增强生成技术", metadata={"source": "b.md"}),
            Document(content="LangGraph是Agent框架", metadata={"source": "c.md"}),
        ]
        retriever.index(docs)
        results = retriever.search("Python编程", top_k=2)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_search_empty_index(self):
        retriever = BM25Retriever()
        results = retriever.search("test", top_k=5)
        assert len(results) == 0

    def test_search_respects_top_k(self):
        retriever = BM25Retriever()
        docs = [Document(content=f"document {i}", metadata={}) for i in range(20)]
        retriever.index(docs)
        results = retriever.search("document", top_k=5)
        assert len(results) <= 5

    def test_search_returns_bm25_score(self):
        retriever = BM25Retriever()
        docs = [
            Document(content="Python programming language", metadata={}),
        ]
        retriever.index(docs)
        results = retriever.search("Python", top_k=1)
        assert len(results) == 1
        assert "bm25_score" in results[0].metadata

    def test_clear(self):
        retriever = BM25Retriever()
        docs = [Document(content="test", metadata={})]
        retriever.index(docs)
        retriever.clear()
        results = retriever.search("test", top_k=5)
        assert len(results) == 0
