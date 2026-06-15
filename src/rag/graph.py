"""LangGraph Agentic RAG 状态机 — 意图路由、查询改写、检索评估循环"""

from __future__ import annotations

import asyncio
import enum
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from src.gateway.client import LLMClient
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.loader import Document
from src.rag.retriever import Retriever

logger = structlog.get_logger(__name__)

MAX_ITERATIONS = 3
RETRIEVE_TOP_K = 20
RERANK_TOP_K = 10
MAX_CHAIN_LENGTH = 5

CHITCHAT_KEYWORDS = frozenset([
    "你好", "hi", "hello", "嗨", "谢谢", "感谢", "bye", "再见",
    "你是谁", "介绍", "早上好", "晚安", "ok", "好的",
])

MATH_PATTERN = re.compile(
    r'(?:计算|算|求|多少|等于|结果)[\s:：]*'
    r'[\d\.\s\+\-\*\/\(\)\（\）\%\^]+'
    r'|[\d\.\s\+\-\*\/\(\)\（\）\%\^]{3,}'
    r'|(?:加|减|乘|除|除以|乘以|加上|减去)[\s]*[\d\.]+'
)

WEATHER_KEYWORDS = frozenset([
    "天气", "气温", "温度", "下雨", "下雪", "晴天", "阴天",
    "weather", "forecast", "今天天气", "明天天气",
])

TOOL_CALL_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ToolProgress:
    progress: int = 0
    message: str = ""
    tool_name: str = ""
    stage: str = "pending"  # pending | running | completed | error


ProgressCallback = Callable[[ToolProgress], Awaitable[None]]


class Intent(enum.Enum):
    CHITCHAT = "chitchat"
    DOC_ANALYSIS = "doc_analysis"
    KNOWLEDGE_QA = "knowledge_qa"
    TOOL_CALL = "tool_call"


class RAGState(TypedDict, total=False):
    query: str
    intent: Intent
    conversation_history: list[dict[str, str]]
    uploaded_doc: str | None
    images: list[str] | None
    rewritten_query: str
    sub_queries: list[str]
    retrieved_docs: list[Document]
    evaluation: str
    answer: str
    sources: list[str]
    iteration: int
    tool_call: dict[str, Any] | None
    tool_result: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_results: list[str] | None
    tool_chain: list[dict[str, Any]] | None
    chain_step_results: dict[str, Any]
    chain_current_step: int


ROUTE_PROMPT = """你是一个意图分类器。根据用户的问题，将其分类为以下三类之一：

- chitchat: 闲聊、问候、通用知识问答（如数学计算、常识、笑话、感谢、告别）、与技术知识库无关的问题
- doc_analysis: 用户上传了文档，要求分析文档内容
- knowledge_qa: 需要从知识库中检索信息来回答的技术问题
  技术主题包括：Python、RAG、LangGraph、机器学习、Docker、FastAPI、向量数据库等

判断标准：
- 如果问题不需要检索特定知识库就能回答 → chitchat
- 如果问题涉及知识库中的技术主题 → knowledge_qa

只返回分类结果，不要返回其他内容。

用户问题: {query}

分类结果:"""

REWRITE_PROMPT = """你是一个查询改写专家。将用户的口语化问题改写为更适合向量检索的专业查询。

重要规则：
1. 如果问题中包含代词（如"它"、"这个"、"那个"、"其"），必须结合对话历史将代词替换为具体实体
2. 将口语化表达转为专业术语
3. 保持查询简洁，适合向量检索

{conversation_context}

原始问题: {query}

改写后的查询（只输出改写结果，不要解释）:"""

EVALUATE_PROMPT = """你是一个检索质量评估专家。判断以下检索到的文档是否足以回答用户问题。

用户问题: {query}

检索到的文档:
{docs}

请判断：
- sufficient: 文档内容足以回答问题
- needs_refinement: 文档相关但不够精确，需要更精确的检索
- needs_decompose: 问题太复杂，需要分解为子问题

只返回评估结果，不要返回其他内容。

评估结果:"""

DECOMPOSE_PROMPT = """你是一个问题分解专家。将以下复杂问题分解为 2-3 个更简单的子问题。

分解原则：
1. 每个子问题应该独立可检索
2. 子问题应覆盖原始问题的所有关键方面
3. 如果问题涉及多个技术/概念，为每个技术/概念单独提问
4. 子问题应该具体，适合向量检索

原始问题: {query}

请返回子问题列表，每行一个，不要编号，不要解释:"""

GENERATE_PROMPT = """你是一个专业的问答助手。根据以下检索到的文档内容回答用户问题。

要求：
1. 基于文档内容回答，不要编造信息
2. 在回答中标注引用来源，使用 [来源N] 格式
3. 如果文档内容不足以回答，明确说明

检索到的文档:
{context}

用户问题: {query}

回答:"""

CHAIN_PLANNING_PROMPT = """你是一个工具调用规划专家。根据用户问题，判断是否需要多步骤工具调用链。

可用工具:
{tools}

用户问题: {query}

规则：
- 如果只需要一个工具调用，返回空数组 []
- 如果需要多个工具按顺序调用（前一步的输出作为后一步的输入），返回调用链
- 使用 "$stepN.result" 引用前一步的输出结果
- 最多 {max_chain} 步

返回调用链 JSON 数组（只返回 JSON，不要解释）:
[
  {{"step": 1, "tool": "工具名", "arguments": {{...}}, "depends_on": []}},
  {{"step": 2, "tool": "工具名", "arguments": {{"param": "$step1.result"}}, "depends_on": [1]}}
]

如果不需要链式调用，返回: []"""


class RAGGraph:
    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever | HybridRetriever,
        tool_registry: Any = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.tool_registry = tool_registry
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph[RAGState]:
        graph = StateGraph(RAGState)

        graph.add_node("route_intent", self._route_intent)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("refine_query", self._refine_query)
        graph.add_node("decompose", self._decompose)
        graph.add_node("generate", self._generate)
        graph.add_node("generate_direct", self._generate_direct)
        graph.add_node("generate_with_context", self._generate_with_context)
        graph.add_node("rag_loop", self._rag_loop)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("generate_with_tool", self._generate_with_tool)

        graph.set_entry_point("route_intent")

        graph.add_conditional_edges(
            "route_intent",
            self._route_decision,
            {
                "chitchat": "generate_direct",
                "doc_analysis": "generate_with_context",
                "knowledge_qa": "rag_loop",
                "tool_call": "execute_tool",
            },
        )

        graph.add_edge("rag_loop", "rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "evaluate")

        graph.add_conditional_edges(
            "evaluate",
            self._evaluate_decision,
            {
                "sufficient": "generate",
                "needs_refinement": "refine_query",
                "needs_decompose": "decompose",
            },
        )

        graph.add_edge("refine_query", "retrieve")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("execute_tool", "generate_with_tool")
        graph.add_edge("generate", END)
        graph.add_edge("generate_direct", END)
        graph.add_edge("generate_with_context", END)
        graph.add_edge("generate_with_tool", END)

        return graph

    def _route_decision(self, state: RAGState) -> str:
        intent = state.get("intent", Intent.KNOWLEDGE_QA)
        if isinstance(intent, Intent):
            return intent.value
        return str(intent)

    def _evaluate_decision(self, state: RAGState) -> str:
        evaluation = state.get("evaluation", "sufficient")
        iteration = state.get("iteration", 0)
        if iteration >= MAX_ITERATIONS:
            return "sufficient"
        return evaluation

    async def _route_intent(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        uploaded_doc = state.get("uploaded_doc")
        images = state.get("images")

        if images:
            logger.info("intent_routed", intent="doc_analysis", reason="images")
            return {"intent": Intent.DOC_ANALYSIS}

        if uploaded_doc:
            logger.info("intent_routed", intent="doc_analysis", reason="uploaded_doc")
            return {"intent": Intent.DOC_ANALYSIS}

        normalized = query.strip().lower()
        if len(normalized) < 20 and any(kw in normalized for kw in CHITCHAT_KEYWORDS):
            logger.info("intent_routed", intent="chitchat", reason="keyword_fast_path")
            return {"intent": Intent.CHITCHAT}

        tool_call, tool_calls = await self._check_tool_call(query)
        if tool_call:
            logger.info("intent_routed", intent="tool_call", tool=tool_call.get("name"))
            return {
                "intent": Intent.TOOL_CALL,
                "tool_call": tool_call,
                "tool_calls": tool_calls,
            }

        start = time.monotonic()
        prompt = ROUTE_PROMPT.format(query=query)
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        elapsed = (time.monotonic() - start) * 1000

        raw = response.content.strip().lower()
        if "chitchat" in raw:
            intent = Intent.CHITCHAT
        elif "doc_analysis" in raw:
            intent = Intent.DOC_ANALYSIS
        else:
            intent = Intent.KNOWLEDGE_QA

        logger.info("intent_routed", intent=intent.value, latency_ms=round(elapsed, 2))
        return {"intent": intent}

    async def _check_tool_call(
        self, query: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not self.tool_registry:
            return None, []

        tools = self.tool_registry.to_openai_functions()
        if not tools:
            return None, []

        direct_match = self._match_tool_by_rules(query, tools)
        if direct_match:
            tool_name = direct_match["name"]
            confidence = direct_match.get("confidence", 1.0)
            logger.info("tool_matched_by_rules", tool=tool_name, confidence=confidence)
            return direct_match, [direct_match]

        try:
            system_prompt = """你是一个工具调用助手。根据用户的问题，选择合适的工具来执行。

当用户需要：
- 计算数学表达式 → 使用 calculator 工具
- 查询天气 → 使用 get_weather 工具
- 统计文本 → 使用 text_count 工具
- 读写文件 → 使用 filesystem 工具
- 查询数据库 → 使用 database 工具
- 搜索网页 → 使用 web_search 工具
- 调用 API → 使用 http_get/http_post 工具
- GitHub 操作（搜索仓库、创建 Issue、查看代码等）→ 使用 github 工具

请直接调用工具，不要自行计算或查询。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.chat(messages, tools=tools)

            if response.tool_calls:
                tool_calls_list = []
                for tc in response.tool_calls:
                    args = tc.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls_list.append({
                        "name": tc.function_name,
                        "arguments": args,
                        "confidence": 0.95,
                    })

                logger.info(
                    "tool_matched_by_function_calling",
                    tools=[tc["name"] for tc in tool_calls_list],
                    count=len(tool_calls_list),
                )
                return tool_calls_list[0], tool_calls_list
        except Exception as e:
            logger.warning("tool_check_failed", error=str(e))

        return None, []

    def _match_tool_by_rules(
        self, query: str, tools: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        normalized = query.strip().lower()
        tool_names = [t.get("function", {}).get("name", "") for t in tools]

        if MATH_PATTERN.search(normalized):
            calculator_tool = next((n for n in tool_names if "calc" in n.lower()), None)
            if calculator_tool:
                expression = self._extract_math_expression(query)
                if expression:
                    return {
                        "name": calculator_tool,
                        "arguments": {"expression": expression},
                        "confidence": 0.95,
                    }

        if any(kw in normalized for kw in WEATHER_KEYWORDS):
            weather_tool = next((n for n in tool_names if "weather" in n.lower()), None)
            if weather_tool:
                city = self._extract_city(query)
                if city:
                    return {
                        "name": weather_tool,
                        "arguments": {"city": city},
                        "confidence": 0.95,
                    }

        return None

    def _extract_math_expression(self, query: str) -> str | None:
        patterns = [
            r'(?:计算|算|求|多少|等于|结果)[\s:：]*(.+?)(?:[？?]|$)',
            r'([\d\.\s\+\-\*\/\(\)\（\）]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                expr = match.group(1).strip()
                expr = re.sub(r'[？?。.]$', '', expr)
                if any(op in expr for op in ['+', '-', '*', '/', '(', '（']):
                    return expr

        cn_map = {
            '加': '+', '减': '-', '乘': '*', '除': '/',
            '乘以': '*', '除以': '/', '加上': '+', '减去': '-'
        }
        for cn, op in cn_map.items():
            if cn in query:
                nums = re.findall(r'[\d\.]+', query)
                if len(nums) >= 2:
                    return f"{nums[0]}{op}{nums[1]}"
        return None

    def _extract_city(self, query: str) -> str | None:
        patterns = [
            r'(?:查询|查|告诉我|看看)?(?:今天|明天|现在)?([\u4e00-\u9fa5]{2,4}?)(?:今天|明天|现在|的)?(?:会|要|有)?(?:天气|气温|温度|下雨|下雪|晴天|阴天)',
        ]

        stop_words = {
            "查询", "查", "告诉", "看看", "今天", "明天", "现在",
            "天气", "气温", "温度", "下雨", "下雪", "会", "要", "有", "晴天", "阴天",
        }

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                city = match.group(1)
                if city not in stop_words and len(city) >= 2:
                    return city
        return None

    def _format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        lines = []
        for i, tool in enumerate(tools, 1):
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            lines.append(f"{i}. 工具名称: {name}")
            lines.append(f"   用途: {desc}")
            if params.get("properties"):
                lines.append("   参数:")
                for pname, pinfo in params["properties"].items():
                    ptype = pinfo.get('type', 'string')
                    pdesc = pinfo.get('description', '')
                    lines.append(f"     - {pname} ({ptype}): {pdesc}")
            lines.append("")
        return "\n".join(lines)

    async def _rewrite_query(self, state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        history = state.get("conversation_history", [])

        conversation_context = ""
        if history:
            recent = history[-6:]
            lines = [f"{m['role']}: {m['content']}" for m in recent]
            conversation_context = "对话历史:\n" + "\n".join(lines)

        start = time.monotonic()
        prompt = REWRITE_PROMPT.format(
            conversation_context=conversation_context,
            query=query,
        )
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        elapsed = (time.monotonic() - start) * 1000
        rewritten = response.content.strip()

        latency = round(elapsed, 2)
        logger.info(
            "query_rewritten",
            original=query[:50],
            rewritten=rewritten[:50],
            latency_ms=latency,
        )
        return {"rewritten_query": rewritten}

    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        sub_queries = state.get("sub_queries", [])

        start = time.monotonic()
        all_docs: list[Document] = []
        if sub_queries:
            for sq in sub_queries:
                docs = self.retriever.retrieve(sq, top_k=RETRIEVE_TOP_K)
                all_docs.extend(docs)
        else:
            all_docs = self.retriever.retrieve(query, top_k=RETRIEVE_TOP_K)

        seen: set[str] = set()
        unique_docs: list[Document] = []
        for doc in all_docs:
            key = doc.content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        reranked = unique_docs[:RERANK_TOP_K]
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "docs_retrieved",
            total=len(all_docs),
            unique=len(unique_docs),
            reranked=len(reranked),
            latency_ms=round(elapsed, 2),
        )
        return {"retrieved_docs": reranked}

    async def _evaluate(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        docs = state.get("retrieved_docs", [])
        iteration = state.get("iteration", 0) + 1

        if not docs:
            return {"evaluation": "needs_decompose", "iteration": iteration}

        start = time.monotonic()
        docs_text = "\n\n".join(
            f"[来源{i+1}] {doc.content[:500]}" for i, doc in enumerate(docs)
        )

        prompt = EVALUATE_PROMPT.format(query=query, docs=docs_text)
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        elapsed = (time.monotonic() - start) * 1000

        raw = response.content.strip().lower()
        if "sufficient" in raw:
            evaluation = "sufficient"
        elif "needs_decompose" in raw:
            evaluation = "needs_decompose"
        else:
            evaluation = "needs_refinement"

        latency = round(elapsed, 2)
        logger.info(
            "evaluation_result",
            evaluation=evaluation,
            iteration=iteration,
            latency_ms=latency,
        )
        return {"evaluation": evaluation, "iteration": iteration}

    async def _refine_query(self, state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        docs = state.get("retrieved_docs", [])

        gaps = []
        for doc in docs[:3]:
            gaps.append(doc.content[:200])

        refined = f"{query}（补充检索：{'、'.join(gaps[:2])}）"

        logger.info("query_refined", refined=refined[:80])
        return {"rewritten_query": refined}

    async def _decompose(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]

        prompt = DECOMPOSE_PROMPT.format(query=query)
        response = await self.llm.chat([{"role": "user", "content": prompt}])

        sub_queries = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ][:3]

        logger.info("query_decomposed", query=query[:50], sub_queries=len(sub_queries))
        return {"sub_queries": sub_queries}

    async def _generate(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        docs = state.get("retrieved_docs", [])

        context = "\n\n".join(
            f"[来源{i+1}] {doc.content}" for i, doc in enumerate(docs)
        )

        prompt = GENERATE_PROMPT.format(context=context, query=query)
        response = await self.llm.chat([{"role": "user", "content": prompt}])

        sources = [
            doc.metadata.get("source", f"来源{i+1}")
            for i, doc in enumerate(docs)
        ]

        logger.info("answer_generated", sources=len(sources))
        return {"answer": response.content, "sources": sources}

    async def _generate_direct(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        history = state.get("conversation_history", [])
        images = state.get("images")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "你是一个友好的AI助手,用自然友好的方式回答问题。"},
        ]
        if history:
            messages.extend(history[-6:])

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": query}]
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": query})

        response = await self.llm.chat(messages)

        logger.info("direct_answer_generated")
        return {"answer": response.content, "sources": []}

    async def _generate_with_context(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        doc_content = state.get("uploaded_doc", "")
        images = state.get("images")

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": query}]
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            context = f"用户上传的文档内容:\n{doc_content}"
            prompt = GENERATE_PROMPT.format(context=context, query=query)
            messages = [{"role": "user", "content": prompt}]

        response = await self.llm.chat(messages)

        logger.info("doc_analysis_generated")
        return {"answer": response.content, "sources": ["uploaded_document"]}

    async def _rag_loop(self, state: RAGState) -> dict[str, Any]:
        return {"iteration": 0, "sub_queries": []}

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        uploaded_doc: str | None = None,
        images: list[str] | None = None,
    ) -> RAGState:
        start = time.monotonic()
        initial_state: RAGState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "uploaded_doc": uploaded_doc,
            "images": images,
            "iteration": 0,
        }

        compiled = self._graph.compile()
        final_state: RAGState = await compiled.ainvoke(initial_state)  # type: ignore[assignment]
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "rag_complete",
            intent=final_state.get("intent", "unknown"),
            iterations=final_state.get("iteration", 0),
            total_latency_ms=round(elapsed, 2),
        )
        return final_state

    async def run_stream(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        uploaded_doc: str | None = None,
        images: list[str] | None = None,
    ) -> Any:
        stream_start = time.monotonic()

        initial_state: RAGState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "uploaded_doc": uploaded_doc,
            "images": images,
            "iteration": 0,
        }

        state: RAGState = dict(initial_state)  # type: ignore[assignment]
        state.update(await self._route_intent(state))  # type: ignore[typeddict-item]

        intent = state.get("intent", Intent.KNOWLEDGE_QA)
        intent_value = intent.value if hasattr(intent, "value") else str(intent)

        if intent_value == "chitchat":
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "你是一个友好的AI助手,用自然友好的方式回答问题。"},
            ]
            if conversation_history:
                messages.extend(conversation_history[-6:])

            if images:
                content: list[dict[str, Any]] = [{"type": "text", "text": query}]
                for img_b64 in images:
                    content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": query})

            prep_ms = (time.monotonic() - stream_start) * 1000
            logger.info("stream_prep_done", intent="chitchat", prep_ms=round(prep_ms, 2))

            async def chitchat_stream() -> Any:
                async for chunk in self.llm.stream_chat(messages):
                    yield chunk

            yield {"type": "intent", "intent": intent_value, "sources": []}
            async for chunk in chitchat_stream():
                yield {"type": "chunk", "content": chunk}
            yield {"type": "done"}
            return

        if intent_value == "doc_analysis" and (uploaded_doc or images):
            context = f"用户上传的文档内容:\n{uploaded_doc}" if uploaded_doc else ""
            prompt = GENERATE_PROMPT.format(context=context, query=query)

            if images:
                msg_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for img_b64 in images:
                    msg_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                messages = [{"role": "user", "content": msg_content}]
            else:
                messages = [{"role": "user", "content": prompt}]

            prep_ms = (time.monotonic() - stream_start) * 1000
            logger.info("stream_prep_done", intent="doc_analysis", prep_ms=round(prep_ms, 2))

            yield {"type": "intent", "intent": intent_value, "sources": ["uploaded_document"]}
            async for chunk in self.llm.stream_chat(messages):
                yield {"type": "chunk", "content": chunk}
            yield {"type": "done"}
            return

        if intent_value == "tool_call":
            tool_call = state.get("tool_call")
            if not tool_call:
                yield {"type": "intent", "intent": intent_value, "sources": []}
                yield {"type": "chunk", "content": "未找到匹配的工具"}
                yield {"type": "done"}
                return

            chain_state = await self._plan_chain(state)
            tool_chain = chain_state.get("tool_chain")

            if tool_chain and len(tool_chain) > 0:
                yield {
                    "type": "progress",
                    "progress": 0,
                    "message": f"规划了 {len(tool_chain)} 步工具调用链",
                    "tool_name": "chain",
                    "stage": "running",
                }

                step_results: dict[str, Any] = {}
                for i, step in enumerate(tool_chain):
                    step_num = step.get("step", i + 1)
                    step_tool = step.get("tool", "")
                    step_args = dict(step.get("arguments", {}))

                    for key, value in step_args.items():
                        if isinstance(value, str) and value.startswith("$"):
                            ref_path = value[1:]
                            ref_step = ref_path.split(".")[0].replace("step", "")
                            ref_key = f"step{ref_step}.result"
                            if ref_key in step_results:
                                step_args[key] = step_results[ref_key]

                    progress = int((i / len(tool_chain)) * 100)
                    yield {
                        "type": "progress",
                        "progress": progress,
                        "message": f"执行步骤 {step_num}/{len(tool_chain)}: {step_tool}",
                        "tool_name": step_tool,
                        "stage": "running",
                    }

                    try:
                        if self.tool_registry:
                            result = await self.tool_registry.call(step_tool, step_args)
                        else:
                            result = "Error: Tool registry not available"
                        step_results[f"step{step_num}.result"] = result
                    except Exception as e:
                        step_results[f"step{step_num}.result"] = f"Error: {e}"

                yield {
                    "type": "progress",
                    "progress": 100,
                    "message": "工具链执行完成",
                    "tool_name": "chain",
                    "stage": "completed",
                }

                results_text_parts = []
                for step in tool_chain:
                    step_num = step.get("step", 0)
                    step_tool = step.get("tool", "unknown")
                    result = step_results.get(f"step{step_num}.result", "无结果")
                    results_text_parts.append(f"步骤{step_num} ({step_tool}): {result}")

                results_text = "\n".join(results_text_parts)
                context = f"工具调用链执行结果:\n{results_text}"

                sources = [f"tool:{step.get('tool', 'unknown')}" for step in tool_chain]
                yield {"type": "intent", "intent": intent_value, "sources": sources}

                gen_prompt = GENERATE_PROMPT.format(context=context, query=query)
                gen_messages = [{"role": "user", "content": gen_prompt}]
                async for chunk in self.llm.stream_chat(gen_messages):
                    yield {"type": "chunk", "content": chunk}
                yield {"type": "done"}
                return

            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            prep_ms = (time.monotonic() - stream_start) * 1000
            logger.info(
                "stream_prep_done",
                intent="tool_call", tool=tool_name,
                prep_ms=round(prep_ms, 2),
            )

            yield {"type": "intent", "intent": intent_value, "sources": [f"tool:{tool_name}"]}

            progress_events: list[dict[str, Any]] = []

            async def on_progress(p: ToolProgress) -> None:
                event = {
                    "type": "progress",
                    "progress": p.progress,
                    "message": p.message,
                    "tool_name": p.tool_name,
                    "stage": p.stage,
                }
                progress_events.append(event)

            yield {
                "type": "progress",
                "progress": 0,
                "message": f"开始调用工具 {tool_name}",
                "tool_name": tool_name,
                "stage": "running",
            }

            tool_result = await self._execute_tool_with_progress(
                tool_name, arguments, on_progress,
            )

            for evt in progress_events:
                yield evt

            context = (
                f"工具调用结果:\n"
                f"工具: {tool_name}\n"
                f"参数: {arguments}\n"
                f"结果: {tool_result}"
            )
            gen_prompt = GENERATE_PROMPT.format(context=context, query=query)
            gen_messages = [{"role": "user", "content": gen_prompt}]

            async for chunk in self.llm.stream_chat(gen_messages):
                yield {"type": "chunk", "content": chunk}
            yield {"type": "done"}
            return

        state.update(await self._rewrite_query(state))  # type: ignore[typeddict-item]
        state.update(await self._retrieve(state))  # type: ignore[typeddict-item]
        state.update(await self._evaluate(state))  # type: ignore[typeddict-item]

        iteration = 0
        while state.get("evaluation") != "sufficient" and iteration < MAX_ITERATIONS:
            if state.get("evaluation") == "needs_decompose":
                state.update(await self._decompose(state))  # type: ignore[typeddict-item]
            else:
                state.update(await self._refine_query(state))  # type: ignore[typeddict-item]
            state.update(await self._retrieve(state))  # type: ignore[typeddict-item]
            state.update(await self._evaluate(state))  # type: ignore[typeddict-item]
            iteration += 1

        docs: list[Document] = state.get("retrieved_docs", [])
        sources = [doc.metadata.get("source", f"来源{i+1}") for i, doc in enumerate(docs)]

        context = "\n\n".join(f"[来源{i+1}] {doc.content}" for i, doc in enumerate(docs))
        prompt = GENERATE_PROMPT.format(context=context, query=query)
        messages = [{"role": "user", "content": prompt}]

        prep_ms = (time.monotonic() - stream_start) * 1000
        logger.info(
            "stream_prep_done",
            intent=intent_value,
            iterations=state.get("iteration", 0),
            prep_ms=round(prep_ms, 2),
        )

        yield {"type": "intent", "intent": intent_value, "sources": sources}
        async for chunk in self.llm.stream_chat(messages):
            yield {"type": "chunk", "content": chunk}
        yield {"type": "done"}

    async def _execute_tool(self, state: RAGState) -> dict[str, Any]:
        tool_calls = state.get("tool_calls") or []

        if not tool_calls:
            single_tool = state.get("tool_call")
            if single_tool:
                tool_calls = [single_tool]
            else:
                return {"tool_result": "Error: No tool call information", "tool_results": []}

        async def _call_single_tool(tool_call: dict[str, Any]) -> str:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            try:
                if self.tool_registry:
                    result = await self.tool_registry.call(tool_name, arguments)
                    logger.info("tool_executed", tool=tool_name, success=True)
                    return str(result)
                else:
                    return "Error: Tool registry not available"
            except Exception as e:
                logger.error("tool_execution_failed", tool=tool_name, error=str(e))
                return f"Error executing tool: {e}"

        tasks = [_call_single_tool(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = tool_calls[i].get("name", "unknown")
                logger.error("tool_execution_failed", tool=tool_name, error=str(result))
                tool_results.append(f"Error executing tool: {result}")
            else:
                tool_results.append(str(result))

        return {
            "tool_result": tool_results[0] if tool_results else "",
            "tool_results": tool_results,
        }

    async def _execute_tool_with_progress(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> str:
        if not self.tool_registry:
            return "Error: Tool registry not available"

        if on_progress:
            await on_progress(ToolProgress(
                progress=10, message=f"准备调用 {tool_name}",
                tool_name=tool_name, stage="running",
            ))

        start = time.monotonic()

        if on_progress:
            await on_progress(ToolProgress(
                progress=30, message=f"正在执行 {tool_name}...",
                tool_name=tool_name, stage="running",
            ))

        try:
            result = await self.tool_registry.call(tool_name, arguments)
            elapsed = (time.monotonic() - start) * 1000

            if on_progress:
                await on_progress(ToolProgress(
                    progress=100,
                    message=f"{tool_name} 执行完成 ({elapsed:.0f}ms)",
                    tool_name=tool_name, stage="completed",
                ))

            logger.info("tool_executed_with_progress", tool=tool_name, latency_ms=round(elapsed, 2))
            return str(result)
        except Exception as e:
            if on_progress:
                await on_progress(ToolProgress(
                    progress=100,
                    message=f"{tool_name} 执行失败: {e}",
                    tool_name=tool_name, stage="error",
                ))

            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            return f"Error executing tool: {e}"

    async def _plan_chain(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        tool_call = state.get("tool_call")

        if not self.tool_registry or not tool_call:
            return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

        tools = self.tool_registry.to_openai_functions()
        if not tools:
            return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

        tools_desc = self._format_tools_for_prompt(tools)

        try:
            prompt = CHAIN_PLANNING_PROMPT.format(
                tools=tools_desc,
                query=query,
                max_chain=MAX_CHAIN_LENGTH,
            )
            response = await self.llm.chat([{"role": "user", "content": prompt}])

            raw = response.content.strip()
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not json_match:
                logger.info("chain_plan_empty", reason="no_json_array")
                return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

            chain = json.loads(json_match.group())

            if not isinstance(chain, list) or len(chain) == 0:
                logger.info("chain_plan_empty", reason="empty_array")
                return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

            if len(chain) > MAX_CHAIN_LENGTH:
                logger.warning("chain_too_long", length=len(chain), max=MAX_CHAIN_LENGTH)
                chain = chain[:MAX_CHAIN_LENGTH]

            if self._detect_circular_deps(chain):
                logger.warning("chain_circular_detected")
                return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

            logger.info("chain_planned", steps=len(chain))
            return {
                "tool_chain": chain,
                "chain_step_results": {},
                "chain_current_step": 0,
                "tool_call": tool_call,
            }
        except Exception as e:
            logger.warning("chain_plan_failed", error=str(e))
            return {"tool_chain": None, "chain_step_results": {}, "chain_current_step": 0}

    def _plan_chain_decision(self, state: RAGState) -> str:
        chain = state.get("tool_chain")
        if chain and len(chain) > 0:
            return "chain"
        return "single_tool"

    async def _execute_chain_step(self, state: RAGState) -> dict[str, Any]:
        chain = state.get("tool_chain", [])
        current_step = state.get("chain_current_step", 0)
        step_results = dict(state.get("chain_step_results", {}))

        if current_step >= len(chain):
            return {"chain_step_results": step_results, "chain_current_step": current_step}

        step = chain[current_step]
        step_num = step.get("step", current_step + 1)
        tool_name = step.get("tool", "")
        arguments = dict(step.get("arguments", {}))

        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                ref_path = value[1:]
                ref_step = ref_path.split(".")[0].replace("step", "")
                ref_key = f"step{ref_step}.result"
                if ref_key in step_results:
                    arguments[key] = step_results[ref_key]
                else:
                    logger.warning("chain_ref_not_found", ref=value, step=step_num)

        try:
            if self.tool_registry:
                result = await self.tool_registry.call(tool_name, arguments)
            else:
                result = "Error: Tool registry not available"

            step_results[f"step{step_num}.result"] = result
            logger.info(
                "chain_step_executed",
                step=step_num,
                tool=tool_name,
                success=True,
                total_steps=len(chain),
            )
        except Exception as e:
            error_msg = f"Error at step {step_num} ({tool_name}): {e}"
            step_results[f"step{step_num}.result"] = error_msg
            logger.error("chain_step_failed", step=step_num, tool=tool_name, error=str(e))

        next_step = current_step + 1
        return {
            "chain_step_results": step_results,
            "chain_current_step": next_step,
        }

    def _chain_step_decision(self, state: RAGState) -> str:
        chain = state.get("tool_chain", [])
        current_step = state.get("chain_current_step", 0)
        if current_step < len(chain):
            return "next_step"
        return "chain_done"

    def _detect_circular_deps(self, chain: list[dict[str, Any]]) -> bool:
        step_deps: dict[int, set[int]] = {}
        for step in chain:
            step_num = step.get("step", 0)
            deps = set(step.get("depends_on", []))
            step_deps[step_num] = deps

        visited: set[int] = set()
        in_stack: set[int] = set()

        def _has_cycle(node: int) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in step_deps.get(node, set()):
                if _has_cycle(dep):
                    return True
            in_stack.discard(node)
            return False

        return any(_has_cycle(step_num) for step_num in step_deps)

    async def _generate_with_chain(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        chain = state.get("tool_chain", [])
        step_results = state.get("chain_step_results", {})

        results_text_parts = []
        for step in chain:
            step_num = step.get("step", 0)
            tool_name = step.get("tool", "unknown")
            result = step_results.get(f"step{step_num}.result", "无结果")
            results_text_parts.append(f"步骤{step_num} ({tool_name}): {result}")

        results_text = "\n".join(results_text_parts)
        context = f"工具调用链执行结果:\n{results_text}"

        prompt = GENERATE_PROMPT.format(context=context, query=query)
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm.chat(messages)

        sources = [f"tool:{step.get('tool', 'unknown')}" for step in chain]

        logger.info("chain_answer_generated", steps=len(chain), sources=len(sources))
        return {
            "answer": response.content,
            "sources": sources,
        }

    async def _generate_with_tool(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        tool_calls = state.get("tool_calls") or []
        tool_results = state.get("tool_results") or []

        if not tool_calls:
            single_tool = state.get("tool_call")
            if single_tool:
                tool_calls = [single_tool]

        if not tool_results:
            single_result = state.get("tool_result")
            if single_result:
                tool_results = [single_result]

        if tool_calls and tool_results and len(tool_calls) == len(tool_results):
            context = "工具调用结果:\n"
            for tc, result in zip(tool_calls, tool_results, strict=False):
                tool_name = tc.get('name', 'unknown')
                tool_args = tc.get('arguments', {})
                context += f"工具: {tool_name}\n参数: {tool_args}\n结果: {result}\n\n"
        elif tool_calls:
            context = "工具调用结果:\n"
            for i, tc in enumerate(tool_calls):
                tool_name = tc.get('name', 'unknown')
                tool_args = tc.get('arguments', {})
                result = tool_results[i] if i < len(tool_results) else "No result"
                context += f"工具: {tool_name}\n参数: {tool_args}\n结果: {result}\n\n"
        else:
            tool_call = state.get("tool_call", {})
            tool_result = state.get("tool_result", "")
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('arguments', {})
            context = (
                f"工具调用结果:\n"
                f"工具: {tool_name}\n"
                f"参数: {tool_args}\n"
                f"结果: {tool_result}"
            )

        prompt = GENERATE_PROMPT.format(context=context, query=query)
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm.chat(messages)

        if tool_calls:
            source_tools = [tc.get("name", "unknown") for tc in tool_calls]
        else:
            default_name = state.get("tool_call", {}).get("name", "unknown")
            source_tools = [default_name]
        logger.info("tool_answer_generated", tools=source_tools)
        return {
            "answer": response.content,
            "sources": [f"tool:{name}" for name in source_tools],
        }
