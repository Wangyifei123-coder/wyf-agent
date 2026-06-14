"""工具注册中心 — schema 定义、权限标注、调用分发"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ToolParameter]
    permissions: list[str] = field(default_factory=list)
    timeout: int = 30
    max_result_length: int = 10000


class Tool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._audit_log: list[dict[str, Any]] = []

    def register(self, tool: Tool) -> None:
        name = tool.schema.name
        self._tools[name] = tool
        logger.info("tool_registered", tool=name, permissions=tool.schema.permissions)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    def to_openai_functions(self) -> list[dict[str, Any]]:
        functions = []
        for tool in self._tools.values():
            params: dict[str, Any] = {}
            required = []
            for p in tool.schema.parameters:
                params[p.name] = {"type": p.type, "description": p.description}
                if p.enum:
                    params[p.name]["enum"] = [str(e) for e in p.enum]
                if p.required:
                    required.append(p.name)

            functions.append({
                "type": "function",
                "function": {
                    "name": tool.schema.name,
                    "description": tool.schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            })
        return functions

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: tool '{name}' not found"

        start = time.monotonic()
        try:
            result = await tool.execute(**arguments)
            if len(result) > tool.schema.max_result_length:
                result = result[: tool.schema.max_result_length] + "\n... [truncated]"

            self._audit(name, arguments, result, time.monotonic() - start, success=True)
            return result

        except Exception as e:
            self._audit(name, arguments, str(e), time.monotonic() - start, success=False)
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return f"Error executing {name}: {e}"

    def _audit(
        self,
        tool: str,
        args: dict[str, Any],
        result: str,
        duration: float,
        success: bool,
    ) -> None:
        entry = {
            "tool": tool,
            "args": args,
            "result_length": len(result),
            "duration_ms": round(duration * 1000, 2),
            "success": success,
            "timestamp": time.time(),
        }
        self._audit_log.append(entry)
        logger.info("tool_call", **entry)
