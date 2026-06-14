"""Tests for src/rag/graph.py"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.gateway.client import LLMResponse
from src.rag.graph import Intent, RAGGraph, RAGState


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_retriever():
    return MagicMock()


def test_initial_state():
    state: RAGState = {
        "query": "test query",
        "intent": Intent.KNOWLEDGE_QA,
        "conversation_history": [],
        "uploaded_doc": None,
        "rewritten_query": "",
        "sub_queries": [],
        "retrieved_docs": [],
        "evaluation": "",
        "answer": "",
        "sources": [],
        "iteration": 0,
    }

    assert state["query"] == "test query"
    assert state["intent"] == Intent.KNOWLEDGE_QA
    assert state["conversation_history"] == []
    assert state["uploaded_doc"] is None
    assert state["rewritten_query"] == ""
    assert state["sub_queries"] == []
    assert state["retrieved_docs"] == []
    assert state["evaluation"] == ""
    assert state["answer"] == ""
    assert state["sources"] == []
    assert state["iteration"] == 0


@pytest.mark.asyncio
async def test_chitchat_bypasses_rag(mock_llm, mock_retriever):
    mock_llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="chitchat",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )
    mock_llm.chat.return_value = LLMResponse(
        content="你好！有什么可以帮你的吗？",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    call_count = 0

    async def mock_chat_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="chitchat",
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        return LLMResponse(
            content="你好！有什么可以帮你的吗？",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    mock_llm.chat = AsyncMock(side_effect=mock_chat_side_effect)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    result = await graph.run("你好呀")

    assert result.get("intent") == Intent.CHITCHAT
    assert result.get("answer") == "你好！有什么可以帮你的吗？"
    assert result.get("sources") == []
    mock_retriever.retrieve.assert_not_called()
