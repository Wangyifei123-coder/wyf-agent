"""工具注册中心 — schema 定义、权限标注、调用分发、输入校验、超时控制、熔断、缓存"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIME = 60
CACHE_DEFAULT_MAX_SIZE = 100
CACHE_DEFAULT_TTL = 300


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
    version: str = "1.0.0"
    permissions: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=lambda: ["admin", "user"])
    timeout: int = 30
    max_result_length: int = 10000
    cacheable: bool = False
    max_retries: int = 0
    retry_delay: float = 1.0


def _parse_version(version: str) -> tuple[int, ...]:
    """将 'major.minor.patch' 解析为可比较的元组。"""
    try:
        return tuple(int(p) for p in version.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


class ToolCache:
    def __init__(
        self,
        max_size: int = CACHE_DEFAULT_MAX_SIZE,
        ttl: int = CACHE_DEFAULT_TTL,
    ) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._hits: int = 0
        self._misses: int = 0
        self._lock = asyncio.Lock()

    def _make_key(self, tool: str, args: dict[str, Any]) -> str:
        data = f"{tool}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()

    async def get(self, tool: str, args: dict[str, Any]) -> str | None:
        async with self._lock:
            key = self._make_key(tool, args)
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    logger.debug("cache_hit", tool=tool, hits=self._hits)
                    return result
                del self._cache[key]
            self._misses += 1
            return None

    async def set(self, tool: str, args: dict[str, Any], result: str) -> None:
        async with self._lock:
            key = self._make_key(tool, args)
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (result, time.time())

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


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
        self._latest_versions: dict[str, str] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cache = ToolCache()

    @staticmethod
    def _make_key(name: str, version: str) -> str:
        return f"{name}:{version}"

    def register(self, tool: Tool) -> None:
        name = tool.schema.name
        version = tool.schema.version
        key = self._make_key(name, version)
        self._tools[key] = tool
        self._circuit_breakers[key] = CircuitBreaker()

        current_latest = self._latest_versions.get(name)
        if current_latest is None or _parse_version(version) > _parse_version(current_latest):
            self._latest_versions[name] = version

        logger.info(
            "tool_registered", tool=name, version=version,
            permissions=tool.schema.permissions,
        )

    def register_mcp_tools(self, mcp_tools: list[Tool]) -> None:
        for tool in mcp_tools:
            self.register(tool)
        logger.info("mcp_tools_registered", count=len(mcp_tools))

    def get(self, name: str, version: str | None = None) -> Tool | None:
        if version:
            return self._tools.get(self._make_key(name, version))
        latest = self._latest_versions.get(name)
        if latest is None:
            return None
        return self._tools.get(self._make_key(name, latest))

    def list_tools(self) -> list[ToolSchema]:
        seen: set[str] = set()
        schemas: list[ToolSchema] = []
        for name, version in self._latest_versions.items():
            tool = self._tools.get(self._make_key(name, version))
            if tool and name not in seen:
                seen.add(name)
                schemas.append(tool.schema)
        return schemas

    def check_permission(self, tool_name: str, user_role: str) -> bool:
        tool = self.get(tool_name)
        if not tool:
            return False

        if not tool.schema.allowed_roles:
            return True

        has_permission = user_role in tool.schema.allowed_roles
        logger.info(
            "permission_check",
            tool=tool_name,
            role=user_role,
            allowed=has_permission,
            allowed_roles=tool.schema.allowed_roles,
        )
        return has_permission

    def list_tools_for_role(self, user_role: str) -> list[ToolSchema]:
        return [
            t.schema for t in self._tools.values()
            if self.check_permission(t.schema.name, user_role)
        ]

    def list_versions(self, name: str) -> list[str]:
        prefix = f"{name}:"
        versions = [
            key[len(prefix):]
            for key in self._tools
            if key.startswith(prefix)
        ]
        versions.sort(key=_parse_version, reverse=True)
        return versions

    def to_openai_functions(self) -> list[dict[str, Any]]:
        functions = []
        seen: set[str] = set()
        for name, version in self._latest_versions.items():
            if name in seen:
                continue
            seen.add(name)
            tool = self._tools.get(self._make_key(name, version))
            if not tool:
                continue

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

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        version: str | None = None,
        user_role: str = "user",
    ) -> str:
        tool = self.get(name, version)
        if not tool:
            ver_msg = f":{version}" if version else ""
            return f"Error: tool '{name}{ver_msg}' not found"

        if not self.check_permission(name, user_role):
            logger.warning("permission_denied", tool=name, role=user_role)
            return f"Error: permission denied for tool '{name}'"
        if not tool:
            ver_msg = f":{version}" if version else ""
            return f"Error: tool '{name}{ver_msg}' not found"

        resolved_version = tool.schema.version
        key = self._make_key(name, resolved_version)

        cb = self._circuit_breakers.get(key)
        if cb and not cb.is_available():
            logger.warning(
                "circuit_breaker_rejected", tool=name, version=resolved_version,
            )
            return (
                f"Error: tool '{name}:{resolved_version}'"
                " is temporarily unavailable (circuit breaker open)"
            )

        try:
            validated_args = validate_arguments(arguments, tool.schema)
        except ValueError as e:
            logger.warning(
                "validation_failed", tool=name,
                version=resolved_version, error=str(e),
            )
            return f"Validation error: {e}"

        if tool.schema.cacheable:
            cached = await self._cache.get(name, validated_args)
            if cached is not None:
                self._audit(
                    name, resolved_version, validated_args, cached,
                    0, success=True, cache_hit=True,
                )
                return cached

        max_retries = tool.schema.max_retries
        retry_delay = tool.schema.retry_delay
        last_error: str | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info(
                    "tool_retry",
                    tool=name,
                    version=resolved_version,
                    attempt=attempt,
                    max_retries=max_retries,
                    delay=delay,
                    error=last_error,
                )
                await asyncio.sleep(delay)

            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    tool.execute(**validated_args),
                    timeout=tool.schema.timeout,
                )
                if len(result) > tool.schema.max_result_length:
                    result = result[: tool.schema.max_result_length] + "\n... [truncated]"

                if tool.schema.cacheable:
                    await self._cache.set(name, validated_args, result)

                if cb:
                    cb.record_success()
                self._audit(
                    name, resolved_version, validated_args, result,
                    time.monotonic() - start, success=True, attempt=attempt,
                )
                return result

            except TimeoutError:
                last_error = f"Timeout after {tool.schema.timeout}s"
                logger.warning(
                    "tool_timeout", tool=name,
                    version=resolved_version, attempt=attempt + 1,
                )
                self._audit(
                    name, resolved_version, validated_args, "Timeout",
                    time.monotonic() - start, success=False, attempt=attempt,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "tool_failed", tool=name,
                    version=resolved_version,
                    attempt=attempt + 1, error=str(e),
                )
                self._audit(
                    name, resolved_version, validated_args, str(e),
                    time.monotonic() - start, success=False, attempt=attempt,
                )

        if cb:
            cb.record_failure()
        logger.error(
            "tool_exhausted_retries", tool=name,
            version=resolved_version,
            attempts=max_retries + 1, error=last_error,
        )
        return (
            f"Error: tool '{name}:{resolved_version}'"
            f" failed after {max_retries + 1} attempt(s): {last_error}"
        )

    def _audit(
        self,
        tool: str,
        version: str,
        args: dict[str, Any],
        result: str,
        duration: float,
        success: bool,
        attempt: int = 0,
        cache_hit: bool = False,
    ) -> None:
        entry = {
            "tool": tool,
            "version": version,
            "args": args,
            "result_length": len(result),
            "duration_ms": round(duration * 1000, 2),
            "success": success,
            "attempt": attempt,
            "cache_hit": cache_hit,
            "timestamp": time.time(),
        }
        self._audit_log.append(entry)
        logger.info("tool_call", **entry)

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._audit_log[-limit:]

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        status = {}
        for key, cb in self._circuit_breakers.items():
            status[key] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
            }
        return status

    def get_cache_stats(self) -> dict[str, Any]:
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        self._cache.clear()
