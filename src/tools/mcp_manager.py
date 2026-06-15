"""MCP 客户端管理器 — 连接 MCP 服务器、发现和调用工具"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

logger = structlog.get_logger(__name__)


class MCPServerConfig:
    def __init__(self, name: str, transport: str, **kwargs: Any) -> None:
        self.name = name
        self.transport = transport
        self.command = kwargs.get("command")
        self.args = kwargs.get("args", [])
        self.env = kwargs.get("env", {})
        self.url = kwargs.get("url")


class MCPManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._configs: dict[str, MCPServerConfig] = {}

    async def connect_server(self, config: MCPServerConfig) -> None:
        try:
            if config.transport == "stdio":
                await self._connect_stdio(config)
            elif config.transport == "http":
                await self._connect_http(config)
            else:
                logger.warning("unsupported_transport", transport=config.transport)
                return

            self._configs[config.name] = config
            await self._discover_tools(config.name)
            logger.info("mcp_server_connected", server=config.name)
        except Exception as e:
            logger.error("mcp_connect_failed", server=config.name, error=str(e))

    async def _connect_stdio(self, config: MCPServerConfig) -> None:
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env or None,
        )
        read, write = await asyncio.wait_for(
            stdio_client(server_params).__aenter__(),
            timeout=30,
        )
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._sessions[config.name] = session

    async def _connect_http(self, config: MCPServerConfig) -> None:
        read, write, _ = await asyncio.wait_for(
            streamable_http_client(config.url).__aenter__(),
            timeout=30,
        )
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._sessions[config.name] = session

    async def _discover_tools(self, server_name: str) -> None:
        session = self._sessions.get(server_name)
        if not session:
            return

        try:
            result = await session.list_tools()
            for tool in result.tools:
                tool_key = f"{server_name}:{tool.name}"
                self._tools[tool_key] = {
                    "name": tool.name,
                    "server": server_name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                }
            logger.info(
                "mcp_tools_discovered",
                server=server_name,
                count=len(result.tools),
            )
        except Exception as e:
            logger.warning("mcp_discover_failed", server=server_name, error=str(e))

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"MCP server '{server_name}' not connected")

        try:
            result = await session.call_tool(tool_name, arguments)
            logger.info(
                "mcp_tool_called",
                server=server_name,
                tool=tool_name,
                success=not result.isError if hasattr(result, 'isError') else True,
            )
            return result
        except Exception as e:
            logger.error(
                "mcp_tool_call_failed",
                server=server_name,
                tool=tool_name,
                error=str(e),
            )
            raise

    def get_all_tools(self) -> list[dict[str, Any]]:
        return list(self._tools.values())

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        tools = []
        for tool in self._tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{tool['server']}_{tool['name']}",
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            })
        return tools

    async def disconnect_all(self) -> None:
        for name, session in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
                logger.info("mcp_server_disconnected", server=name)
            except Exception as e:
                logger.warning("mcp_disconnect_failed", server=name, error=str(e))
        self._sessions.clear()
        self._tools.clear()

    @staticmethod
    def load_config(config_path: str) -> list[MCPServerConfig]:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        configs = []
        for server in data.get("servers", []):
            configs.append(MCPServerConfig(
                name=server["name"],
                transport=server["transport"],
                command=server.get("command"),
                args=server.get("args", []),
                env=server.get("env", {}),
                url=server.get("url"),
            ))
        return configs
