"""Tests for src/tools/knowledge.py"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from src.tools.knowledge import KnowledgeSearchTool


@pytest.fixture
def mock_graph():
    graph = AsyncMock()
    graph.run = AsyncMock(return_value={
        "answer": "这是一个测试回答",
        "sources": ["doc1.md", "doc2.md"],
    })
    return graph


@pytest.fixture
def tool(mock_graph):
    return KnowledgeSearchTool(mock_graph)


class TestSchema:
    def test_name(self, tool):
        assert tool.schema.name == "search_knowledge_base"

    def test_parameters_has_query(self, tool):
        param_names = [p.name for p in tool.schema.parameters]
        assert "query" in param_names


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(self, tool):
        result = await tool.execute(query="测试问题")
        assert "这是一个测试回答" in result
        assert "doc1.md" in result
        assert "doc2.md" in result

    @pytest.mark.asyncio
    async def test_calls_graph_with_query(self, tool, mock_graph):
        await tool.execute(query="什么是RAG")
        mock_graph.run.assert_called_once_with("什么是RAG")

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, tool):
        result = await tool.execute(query="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_no_sources_in_response(self, mock_graph):
        mock_graph.run = AsyncMock(return_value={"answer": "只有答案"})
        tool = KnowledgeSearchTool(mock_graph)
        result = await tool.execute(query="test")
        assert "只有答案" in result
        assert "引用来源" not in result
