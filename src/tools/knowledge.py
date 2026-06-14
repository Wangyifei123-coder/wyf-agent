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
                ToolParameter(name="query", type="string", description="用户的检索问题", required=True),
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
