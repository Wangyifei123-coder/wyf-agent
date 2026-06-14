"""Tests for src/rag/retriever.py"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.rag.loader import Document
from src.rag.retriever import Retriever, rrf_score


def test_single_rank():
    assert rrf_score([1], k=60) == 1.0 / 61


def test_multiple_ranks():
    assert rrf_score([1, 3], k=60) == 1.0 / 61 + 1.0 / 63


def test_empty_ranks():
    assert rrf_score([], k=60) == 0


def test_retrieve_returns_top_k():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        Document(content="doc A", metadata={}),
        Document(content="doc B", metadata={}),
        Document(content="doc C", metadata={}),
    ]
    retriever = Retriever(mock_store)
    results = retriever.retrieve("test query", top_k=2)
    assert len(results) <= 2


def test_retrieve_calls_vector_store():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        Document(content="hello world", metadata={}),
    ]
    retriever = Retriever(mock_store)
    retriever.retrieve("hello", top_k=5)
    mock_store.search.assert_called_once_with("hello", top_k=10)
