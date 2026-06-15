"""MCP 工具适配器 — 将 MCP 工具包装为 Tool 接口"""

from __future__ import annotations

from typing import Any

from .mcp_manager import MCPManager
from .registry import Tool, ToolParameter, ToolSchema


class MCPToolAdapter(Tool):
    def __init__(self, tool_info: dict[str, Any], manager: MCPManager) -> None:
        self._info = tool_info
        self._manager = manager
        self._schema = self._build_schema()

    def _build_schema(self) -> ToolSchema:
        params = []
        input_schema = self._info.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
                enum=prop.get("enum"),
            ))

        return ToolSchema(
            name=f"mcp_{self._info['server']}_{self._info['name']}",
            description=self._info["description"],
            parameters=params,
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, **kwargs: Any) -> str:
        result = await self._manager.call_tool(
            self._info["server"],
            self._info["name"],
            kwargs,
        )

        if hasattr(result, 'content') and result.content:
            texts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    texts.append(content.text)
            return "\n".join(texts) if texts else str(result)

        return str(result)
