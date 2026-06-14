"""LangGraph Agentic RAG 状态机 — 意图路由、查询改写、检索评估循环"""

from __future__ import annotations

import enum
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


class Intent(enum.Enum):
    CHITCHAT = "chitchat"
    DOC_ANALYSIS = "doc_analysis"
    KNOWLEDGE_QA = "knowledge_qa"


class RAGState(TypedDict, total=False):
    query: str
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


ROUTE_PROMPT = """你是一个意图分类器。根据用户的问题，将其分类为以下三类之一：

- chitchat: 闲聊、问候、与知识无关的对话
- doc_analysis: 用户上传了文档，要求分析文档内容
- knowledge_qa: 需要从知识库中检索信息来回答的问题

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


class RAGGraph:
    def __init__(self, llm: LLMClient, retriever: Retriever | HybridRetriever) -> None:
        self.llm = llm
        self.retriever = retriever
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

        graph.set_entry_point("route_intent")

        graph.add_conditional_edges(
            "route_intent",
            self._route_decision,
            {
                "chitchat": "generate_direct",
                "doc_analysis": "generate_with_context",
                "knowledge_qa": "rag_loop",
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
        graph.add_edge("generate", END)
        graph.add_edge("generate_direct", END)
        graph.add_edge("generate_with_context", END)

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

        if uploaded_doc:
            logger.info("intent_routed", intent="doc_analysis", reason="uploaded_doc")
            return {"intent": Intent.DOC_ANALYSIS}

        prompt = ROUTE_PROMPT.format(query=query)
        response = await self.llm.chat([{"role": "user", "content": prompt}])

        raw = response.content.strip().lower()
        if "chitchat" in raw:
            intent = Intent.CHITCHAT
        elif "doc_analysis" in raw:
            intent = Intent.DOC_ANALYSIS
        else:
            intent = Intent.KNOWLEDGE_QA

        logger.info("intent_routed", intent=intent.value)
        return {"intent": intent}

    async def _rewrite_query(self, state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        history = state.get("conversation_history", [])

        conversation_context = ""
        if history:
            recent = history[-6:]
            lines = [f"{m['role']}: {m['content']}" for m in recent]
            conversation_context = "对话历史:\n" + "\n".join(lines)

        prompt = REWRITE_PROMPT.format(
            conversation_context=conversation_context,
            query=query,
        )
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        rewritten = response.content.strip()

        logger.info("query_rewritten", original=query[:50], rewritten=rewritten[:50])
        return {"rewritten_query": rewritten}

    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        sub_queries = state.get("sub_queries", [])

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

        logger.info(
            "docs_retrieved",
            total=len(all_docs),
            unique=len(unique_docs),
            reranked=len(reranked),
        )
        return {"retrieved_docs": reranked}

    async def _evaluate(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        docs = state.get("retrieved_docs", [])
        iteration = state.get("iteration", 0) + 1

        if not docs:
            return {"evaluation": "needs_decompose", "iteration": iteration}

        docs_text = "\n\n".join(
            f"[来源{i+1}] {doc.content[:500]}" for i, doc in enumerate(docs)
        )

        prompt = EVALUATE_PROMPT.format(query=query, docs=docs_text)
        response = await self.llm.chat([{"role": "user", "content": prompt}])

        raw = response.content.strip().lower()
        if "sufficient" in raw:
            evaluation = "sufficient"
        elif "needs_decompose" in raw:
            evaluation = "needs_decompose"
        else:
            evaluation = "needs_refinement"

        logger.info("evaluation_result", evaluation=evaluation, iteration=iteration)
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

        messages: list[dict[str, str]] = []
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": query})

        response = await self.llm.chat(messages)

        logger.info("direct_answer_generated")
        return {"answer": response.content, "sources": []}

    async def _generate_with_context(self, state: RAGState) -> dict[str, Any]:
        query = state["query"]
        doc_content = state.get("uploaded_doc", "")

        context = f"用户上传的文档内容:\n{doc_content}"

        prompt = GENERATE_PROMPT.format(context=context, query=query)
        response = await self.llm.chat([{"role": "user", "content": prompt}])

        logger.info("doc_analysis_generated")
        return {"answer": response.content, "sources": ["uploaded_document"]}

    async def _rag_loop(self, state: RAGState) -> dict[str, Any]:
        return {"iteration": 0, "sub_queries": []}

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        uploaded_doc: str | None = None,
    ) -> RAGState:
        initial_state: RAGState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "uploaded_doc": uploaded_doc,
            "iteration": 0,
        }

        compiled = self._graph.compile()
        final_state: RAGState = await compiled.ainvoke(initial_state)  # type: ignore[assignment]

        logger.info(
            "rag_complete",
            intent=final_state.get("intent", "unknown"),
            iterations=final_state.get("iteration", 0),
        )
        return final_state
