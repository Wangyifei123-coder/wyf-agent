"""Tests for VectorStore (ChromaDB wrapper)."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

from src.rag.loader import Document
from src.rag.vectorstore import VectorStore


@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.embed_query.return_value = [0.1, 0.2, 0.3]
    return service


@pytest.fixture
def vectorstore(mock_embedding_service):
    tmpdir = tempfile.mkdtemp()
    store = VectorStore(
        collection_name="test_collection",
        embedding_service=mock_embedding_service,
        persist_directory=tmpdir,
    )
    yield store
    del store._client


def test_add_and_search(vectorstore, mock_embedding_service):
    docs = [
        Document(content="Python is a programming language", metadata={"topic": "python"}),
        Document(content="The cat sat on the mat", metadata={"topic": "animals"}),
    ]
    mock_embedding_service.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    vectorstore.add_documents(docs)
    mock_embedding_service.embed_texts.assert_called_once()

    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embedding_service.embed_query.reset_mock()
    results = vectorstore.search("programming", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)


def test_search_returns_metadata(vectorstore, mock_embedding_service):
    docs = [
        Document(content="Test document", metadata={"source": "test.md", "page": 1}),
    ]
    mock_embedding_service.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    vectorstore.add_documents(docs)

    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    results = vectorstore.search("test", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "test.md"
    assert results[0].metadata["page"] == 1


def test_empty_search(vectorstore, mock_embedding_service):
    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    results = vectorstore.search("anything", top_k=5)
    assert results == []


def test_get_collection_stats(vectorstore):
    stats = vectorstore.get_collection_stats()
    assert stats["count"] == 0
    assert stats["name"] == "test_collection"
