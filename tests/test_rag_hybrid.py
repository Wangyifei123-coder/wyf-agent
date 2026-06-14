"""混合检索器测试"""

from unittest.mock import MagicMock
from src.rag.bm25_retriever import BM25Retriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.loader import Document
from src.rag.retriever import Retriever


class TestHybridRetriever:
    def test_retrieve_returns_results(self):
        mock_vector = MagicMock(spec=Retriever)
        mock_vector.retrieve.return_value = [
            Document(content="Python是编程语言", metadata={"source": "a.md"}),
        ]
        bm25 = BM25Retriever()
        bm25.index([
            Document(content="Python是编程语言", metadata={"source": "a.md"}),
            Document(content="RAG是检索技术", metadata={"source": "b.md"}),
        ])

        hybrid = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=bm25,
        )
        results = hybrid.retrieve("Python", top_k=5)
        assert len(results) >= 1

    def test_retrieve_deduplicates(self):
        mock_vector = MagicMock(spec=Retriever)
        mock_vector.retrieve.return_value = [
            Document(content="Python是编程语言", metadata={"source": "a.md"}),
        ]
        bm25 = BM25Retriever()
        bm25.index([
            Document(content="Python是编程语言", metadata={"source": "a.md"}),
        ])

        hybrid = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=bm25,
        )
        results = hybrid.retrieve("Python", top_k=5)
        assert len(results) == 1

    def test_retrieve_respects_top_k(self):
        mock_vector = MagicMock(spec=Retriever)
        mock_vector.retrieve.return_value = [
            Document(content=f"doc {i}", metadata={}) for i in range(10)
        ]
        bm25 = BM25Retriever()
        bm25.index([Document(content=f"doc {i}", metadata={}) for i in range(10)])

        hybrid = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=bm25,
        )
        results = hybrid.retrieve("doc", top_k=3)
        assert len(results) <= 3

    def test_index_calls_bm25(self):
        mock_vector = MagicMock(spec=Retriever)
        bm25 = BM25Retriever()
        hybrid = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=bm25,
        )
        docs = [Document(content="test", metadata={})]
        hybrid.index(docs)
        assert len(bm25._documents) == 1
