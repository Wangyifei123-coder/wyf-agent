"""MCP 客户端管理器 — 连接 MCP 服务器、发现和调用工具"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

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


class MCPServerConnection:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: ClientSession | None = None
        self._context: Any = None

    async def connect(self) -> None:
        if self.config.transport == "stdio":
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env or None,
            )
            self._context = stdio_client(server_params)
            read, write = await self._context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            await self.session.initialize()
        elif self.config.transport == "http":
            self._context = streamable_http_client(self.config.url)
            read, write, _ = await self._context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            await self.session.initialize()

    async def disconnect(self) -> None:
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
        if self._context:
            try:
                await self._context.__aexit__(None, None, None)
            except Exception:
                pass


class MCPManager:
    def __init__(self) -> None:
        self._connections: dict[str, MCPServerConnection] = {}
        self._tools: dict[str, dict[str, Any]] = {}

    async def connect_server(self, config: MCPServerConfig) -> None:
        try:
            conn = MCPServerConnection(config)
            await conn.connect()
            self._connections[config.name] = conn
            await self._discover_tools(config.name)
            logger.info("mcp_server_connected", server=config.name)
        except Exception as e:
            logger.error("mcp_connect_failed", server=config.name, error=str(e))

    async def _discover_tools(self, server_name: str) -> None:
        conn = self._connections.get(server_name)
        if not conn or not conn.session:
            return

        try:
            result = await conn.session.list_tools()
            for tool in result.tools:
                tool_key = f"{server_name}:{tool.name}"
                self._tools[tool_key] = {
                    "name": tool.name,
                    "server": server_name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
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
        conn = self._connections.get(server_name)
        if not conn or not conn.session:
            raise ValueError(f"MCP server '{server_name}' not connected")

        result = await conn.session.call_tool(tool_name, arguments)
        logger.info("mcp_tool_called", server=server_name, tool=tool_name)
        return result

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
        for name, conn in self._connections.items():
            try:
                await conn.disconnect()
                logger.info("mcp_server_disconnected", server=name)
            except Exception as e:
                logger.warning("mcp_disconnect_failed", server=name, error=str(e))
        self._connections.clear()
        self._tools.clear()

    @staticmethod
    def load_config(config_path: str) -> list[MCPServerConfig]:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        configs = []
        for server in data.get("servers", []):
            if server.get("enabled", True):
                configs.append(MCPServerConfig(
                    name=server["name"],
                    transport=server["transport"],
                    command=server.get("command"),
                    args=server.get("args", []),
                    env=server.get("env", {}),
                    url=server.get("url"),
                ))
        return configs

    @staticmethod
    def save_config(config_path: str, servers: list[dict[str, Any]]) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"servers": servers}, f, default_flow_style=False)

    def get_connected_servers(self) -> list[str]:
        return list(self._connections.keys())
