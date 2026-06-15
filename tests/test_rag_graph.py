"""Tests for src/rag/graph.py"""
from __future__ import annotations

import asyncio
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
            content="你好！有什么可以帮你的吗？",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    result = await graph.run("你好呀")

    assert result.get("intent") == Intent.CHITCHAT
    assert result.get("answer") == "你好！有什么可以帮你的吗？"
    assert result.get("sources") == []
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_chitchat_llm_routing(mock_llm, mock_retriever):
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
            content="当然可以！",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    mock_llm.chat = AsyncMock(side_effect=mock_chat_side_effect)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    result = await graph.run("今天天气怎么样，适合出去玩吗？")

    assert result.get("intent") == Intent.CHITCHAT
    assert result.get("answer") == "当然可以！"
    assert result.get("sources") == []
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_tool_execution(mock_llm, mock_retriever):
    """Test parallel execution of multiple tool calls."""
    mock_tool_registry = MagicMock()
    
    call_log = []
    
    async def mock_tool_call(tool_name, arguments):
        call_log.append({"tool": tool_name, "args": arguments, "time": asyncio.get_event_loop().time()})
        if tool_name == "calculator":
            await asyncio.sleep(0.1)
            return "42"
        elif tool_name == "get_weather":
            await asyncio.sleep(0.1)
            return "晴天，25°C"
        return "unknown tool"
    
    mock_tool_registry.call = AsyncMock(side_effect=mock_tool_call)
    mock_tool_registry.to_openai_functions.return_value = [
        {"function": {"name": "calculator", "description": "计算数学表达式"}},
        {"function": {"name": "get_weather", "description": "查询天气"}},
    ]
    
    mock_llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="根据计算结果是42，天气是晴天25°C",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
    )
    
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)
    
    state: RAGState = {
        "query": "测试查询",
        "tool_calls": [
            {"name": "calculator", "arguments": {"expression": "6*7"}},
            {"name": "get_weather", "arguments": {"city": "北京"}},
        ],
        "tool_call": {"name": "calculator", "arguments": {"expression": "6*7"}},
    }
    
    result = await graph._execute_tool(state)
    
    assert "tool_results" in result
    assert len(result["tool_results"]) == 2
    assert result["tool_results"][0] == "42"
    assert result["tool_results"][1] == "晴天，25°C"
    
    assert len(call_log) == 2
    assert call_log[0]["tool"] == "calculator"
    assert call_log[1]["tool"] == "get_weather"


@pytest.mark.asyncio
async def test_single_tool_backward_compatibility(mock_llm, mock_retriever):
    """Test backward compatibility with single tool call."""
    mock_tool_registry = MagicMock()
    
    async def mock_tool_call(tool_name, arguments):
        if tool_name == "calculator":
            return "42"
        return "unknown"
    
    mock_tool_registry.call = AsyncMock(side_effect=mock_tool_call)
    
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)
    
    state: RAGState = {
        "query": "测试查询",
        "tool_call": {"name": "calculator", "arguments": {"expression": "6*7"}},
    }
    
    result = await graph._execute_tool(state)
    
    assert "tool_result" in result
    assert result["tool_result"] == "42"
    assert "tool_results" in result
    assert result["tool_results"] == ["42"]


@pytest.mark.asyncio
async def test_tool_execution_error_handling(mock_llm, mock_retriever):
    """Test that one tool failure doesn't affect others."""
    mock_tool_registry = MagicMock()
    
    async def mock_tool_call(tool_name, arguments):
        if tool_name == "calculator":
            return "42"
        elif tool_name == "get_weather":
            raise Exception("API 超时")
        return "unknown"
    
    mock_tool_registry.call = AsyncMock(side_effect=mock_tool_call)
    
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)
    
    state: RAGState = {
        "query": "测试查询",
        "tool_calls": [
            {"name": "calculator", "arguments": {"expression": "6*7"}},
            {"name": "get_weather", "arguments": {"city": "北京"}},
        ],
        "tool_call": {"name": "calculator", "arguments": {"expression": "6*7"}},
    }
    
    result = await graph._execute_tool(state)
    
    assert len(result["tool_results"]) == 2
    assert result["tool_results"][0] == "42"
    assert "Error executing tool" in result["tool_results"][1]


@pytest.mark.asyncio
async def test_plan_chain_returns_chain(mock_llm, mock_retriever):
    """Test _plan_chain returns a chain when LLM returns valid chain JSON."""
    mock_tool_registry = MagicMock()

    mock_tool_registry.to_openai_functions.return_value = [
        {
            "function": {
                "name": "web_search",
                "description": "搜索网页",
                "parameters": {"properties": {"query": {"type": "string"}}},
            }
        },
        {
            "function": {
                "name": "read_url",
                "description": "读取URL内容",
                "parameters": {"properties": {"url": {"type": "string"}}},
            }
        },
    ]

    chain_json = '''[
        {"step": 1, "tool": "web_search", "arguments": {"query": "Python教程"}, "depends_on": []},
        {"step": 2, "tool": "read_url", "arguments": {"url": "$step1.result"}, "depends_on": [1]}
    ]'''

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(content=chain_json, model="test", usage={})
        return LLMResponse(content="answer", model="test", usage={})

    mock_llm.chat = AsyncMock(side_effect=mock_chat)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)

    state: RAGState = {
        "query": "搜索Python教程并读取内容",
        "tool_call": {"name": "web_search", "arguments": {"query": "Python教程"}},
    }

    result = await graph._plan_chain(state)

    assert result["tool_chain"] is not None
    assert len(result["tool_chain"]) == 2
    assert result["chain_current_step"] == 0
    assert result["chain_step_results"] == {}


@pytest.mark.asyncio
async def test_plan_chain_returns_none_for_empty(mock_llm, mock_retriever):
    """Test _plan_chain returns None chain when LLM returns empty array."""
    mock_tool_registry = MagicMock()

    mock_tool_registry.to_openai_functions.return_value = [
        {
            "function": {
                "name": "calculator",
                "description": "计算",
                "parameters": {"properties": {}},
            }
        },
    ]

    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return LLMResponse(content="[]", model="test", usage={})

    mock_llm.chat = AsyncMock(side_effect=mock_chat)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)

    state: RAGState = {
        "query": "计算1+1",
        "tool_call": {"name": "calculator", "arguments": {"expression": "1+1"}},
    }

    result = await graph._plan_chain(state)

    assert result["tool_chain"] is None


@pytest.mark.asyncio
async def test_execute_chain_step_single_step(mock_llm, mock_retriever):
    """Test executing a single step chain."""
    mock_tool_registry = MagicMock()

    async def mock_call(tool_name, arguments):
        return "result_value"

    mock_tool_registry.call = AsyncMock(side_effect=mock_call)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)

    state: RAGState = {
        "query": "test",
        "tool_chain": [
            {"step": 1, "tool": "calculator", "arguments": {"expression": "1+1"}, "depends_on": []},
        ],
        "chain_step_results": {},
        "chain_current_step": 0,
    }

    result = await graph._execute_chain_step(state)

    assert result["chain_current_step"] == 1
    assert result["chain_step_results"]["step1.result"] == "result_value"


@pytest.mark.asyncio
async def test_execute_chain_step_with_dependency(mock_llm, mock_retriever):
    """Test chain step correctly resolves $stepN.result references."""
    mock_tool_registry = MagicMock()

    async def mock_call(tool_name, arguments):
        if tool_name == "web_search":
            return "https://example.com"
        elif tool_name == "read_url":
            return f"Content from {arguments.get('url', 'unknown')}"
        return "unknown"

    mock_tool_registry.call = AsyncMock(side_effect=mock_call)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)

    state: RAGState = {
        "query": "test",
        "tool_chain": [
            {
                "step": 1,
                "tool": "web_search",
                "arguments": {"query": "test"},
                "depends_on": [],
            },
            {
                "step": 2,
                "tool": "read_url",
                "arguments": {"url": "$step1.result"},
                "depends_on": [1],
            },
        ],
        "chain_step_results": {"step1.result": "https://example.com"},
        "chain_current_step": 1,
    }

    result = await graph._execute_chain_step(state)

    assert result["chain_current_step"] == 2
    assert result["chain_step_results"]["step2.result"] == "Content from https://example.com"
    mock_tool_registry.call.assert_called_once_with("read_url", {"url": "https://example.com"})


@pytest.mark.asyncio
async def test_execute_chain_step_error_continues(mock_llm, mock_retriever):
    """Test chain records error but continues to next step."""
    mock_tool_registry = MagicMock()

    call_count = 0

    async def mock_call(tool_name, arguments):
        nonlocal call_count
        call_count += 1
        if tool_name == "failing_tool":
            raise Exception("tool failed")
        return "ok"

    mock_tool_registry.call = AsyncMock(side_effect=mock_call)

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever, tool_registry=mock_tool_registry)

    state: RAGState = {
        "query": "test",
        "tool_chain": [
            {"step": 1, "tool": "failing_tool", "arguments": {}, "depends_on": []},
            {"step": 2, "tool": "ok_tool", "arguments": {}, "depends_on": []},
        ],
        "chain_step_results": {},
        "chain_current_step": 0,
    }

    result = await graph._execute_chain_step(state)

    assert result["chain_current_step"] == 1
    assert "Error at step 1" in result["chain_step_results"]["step1.result"]


def test_chain_step_decision_next_step(mock_llm, mock_retriever):
    """Test _chain_step_decision returns next_step when more steps remain."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    state: RAGState = {
        "query": "test",
        "tool_chain": [
            {"step": 1, "tool": "a", "arguments": {}},
            {"step": 2, "tool": "b", "arguments": {}},
        ],
        "chain_current_step": 1,
    }

    assert graph._chain_step_decision(state) == "next_step"


def test_chain_step_decision_done(mock_llm, mock_retriever):
    """Test _chain_step_decision returns chain_done when all steps executed."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    state: RAGState = {
        "query": "test",
        "tool_chain": [
            {"step": 1, "tool": "a", "arguments": {}},
        ],
        "chain_current_step": 1,
    }

    assert graph._chain_step_decision(state) == "chain_done"


def test_detect_circular_deps_no_cycle(mock_llm, mock_retriever):
    """Test _detect_circular_deps returns False for valid chain."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    chain = [
        {"step": 1, "tool": "a", "arguments": {}, "depends_on": []},
        {"step": 2, "tool": "b", "arguments": {}, "depends_on": [1]},
        {"step": 3, "tool": "c", "arguments": {}, "depends_on": [2]},
    ]

    assert graph._detect_circular_deps(chain) is False


def test_detect_circular_deps_with_cycle(mock_llm, mock_retriever):
    """Test _detect_circular_deps returns True for circular chain."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    chain = [
        {"step": 1, "tool": "a", "arguments": {}, "depends_on": [2]},
        {"step": 2, "tool": "b", "arguments": {}, "depends_on": [1]},
    ]

    assert graph._detect_circular_deps(chain) is True


def test_plan_chain_decision_single_tool(mock_llm, mock_retriever):
    """Test _plan_chain_decision routes to single_tool when no chain."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    state: RAGState = {"query": "test", "tool_chain": None}
    assert graph._plan_chain_decision(state) == "single_tool"

    state2: RAGState = {"query": "test", "tool_chain": []}
    assert graph._plan_chain_decision(state2) == "single_tool"


def test_plan_chain_decision_chain(mock_llm, mock_retriever):
    """Test _plan_chain_decision routes to chain when chain exists."""
    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    state: RAGState = {
        "query": "test",
        "tool_chain": [{"step": 1, "tool": "a", "arguments": {}}],
    }
    assert graph._plan_chain_decision(state) == "chain"


@pytest.mark.asyncio
async def test_generate_with_chain(mock_llm, mock_retriever):
    """Test _generate_with_chain formats chain results and generates answer."""
    mock_llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="这是基于链结果的回答",
            model="test",
            usage={},
        )
    )

    graph = RAGGraph(llm=mock_llm, retriever=mock_retriever)

    state: RAGState = {
        "query": "test query",
        "tool_chain": [
            {"step": 1, "tool": "web_search", "arguments": {}},
            {"step": 2, "tool": "read_url", "arguments": {}},
        ],
        "chain_step_results": {
            "step1.result": "https://example.com",
            "step2.result": "page content",
        },
    }

    result = await graph._generate_with_chain(state)

    assert result["answer"] == "这是基于链结果的回答"
    assert len(result["sources"]) == 2
    assert "tool:web_search" in result["sources"]
    assert "tool:read_url" in result["sources"]
