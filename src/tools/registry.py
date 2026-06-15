"""工具注册中心 — schema 定义、权限标注、调用分发、输入校验、超时控制、熔断"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIME = 60


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


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        reset_time: float = CIRCUIT_BREAKER_RESET_TIME,
    ) -> None:
        self.threshold = threshold
        self.reset_time = reset_time
        self.failure_count: int = 0
        self.last_failure_time: float = 0
        self.state: str = "closed"

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning("circuit_breaker_open", failures=self.failure_count)

    def is_available(self) -> bool:
        if self.state == "closed":
            return True
        if time.time() - self.last_failure_time > self.reset_time:
            self.state = "half_open"
            return True
        return False


def validate_arguments(arguments: dict[str, Any], schema: ToolSchema) -> dict[str, Any]:
    validated = {}
    param_map = {p.name: p for p in schema.parameters}

    for p in schema.parameters:
        if p.required and p.name not in arguments:
            raise ValueError(f"Missing required parameter: {p.name}")

    for name, value in arguments.items():
        if name not in param_map:
            continue

        param = param_map[name]

        if param.type == "integer":
            try:
                validated[name] = int(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Parameter '{name}' must be an integer") from e
        elif param.type == "number":
            try:
                validated[name] = float(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Parameter '{name}' must be a number") from e
        elif param.type == "boolean":
            if isinstance(value, str):
                validated[name] = value.lower() in ("true", "1", "yes")
            else:
                validated[name] = bool(value)
        elif param.type == "string":
            validated[name] = str(value)
        else:
            validated[name] = value

        if param.enum and validated.get(name) not in param.enum:
            raise ValueError(f"Parameter '{name}' must be one of {param.enum}")

    return validated


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def register(self, tool: Tool) -> None:
        name = tool.schema.name
        self._tools[name] = tool
        self._circuit_breakers[name] = CircuitBreaker()
        logger.info("tool_registered", tool=name, permissions=tool.schema.permissions)

    def register_mcp_tools(self, mcp_tools: list[Tool]) -> None:
        for tool in mcp_tools:
            self.register(tool)
        logger.info("mcp_tools_registered", count=len(mcp_tools))

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

        cb = self._circuit_breakers.get(name)
        if cb and not cb.is_available():
            logger.warning("circuit_breaker_rejected", tool=name)
            return f"Error: tool '{name}' is temporarily unavailable (circuit breaker open)"

        try:
            validated_args = validate_arguments(arguments, tool.schema)
        except ValueError as e:
            logger.warning("validation_failed", tool=name, error=str(e))
            return f"Validation error: {e}"

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.execute(**validated_args),
                timeout=tool.schema.timeout,
            )
            if len(result) > tool.schema.max_result_length:
                result = result[: tool.schema.max_result_length] + "\n... [truncated]"

            if cb:
                cb.record_success()
            self._audit(name, validated_args, result, time.monotonic() - start, success=True)
            return result

        except TimeoutError:
            if cb:
                cb.record_failure()
            self._audit(name, validated_args, "Timeout", time.monotonic() - start, success=False)
            logger.error("tool_timeout", tool=name, timeout=tool.schema.timeout)
            return f"Error: tool '{name}' timed out after {tool.schema.timeout}s"

        except Exception as e:
            if cb:
                cb.record_failure()
            self._audit(name, validated_args, str(e), time.monotonic() - start, success=False)
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

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._audit_log[-limit:]

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        status = {}
        for name, cb in self._circuit_breakers.items():
            status[name] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
            }
        return status
