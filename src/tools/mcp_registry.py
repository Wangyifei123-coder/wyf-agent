"""MCP Registry 客户端 — 搜索和发现 MCP 服务器"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

MCP_REGISTRY_URL = "https://registry.modelcontextprotocol.io"
GITHUB_API_URL = "https://api.github.com"


class MCPRegistryClient:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            results = await self._search_github(query, limit)
            if results:
                return results
        except Exception as e:
            logger.warning("github_search_failed", error=str(e))

        return await self._search_npm(query, limit)

    async def _search_github(self, query: str, limit: int) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{GITHUB_API_URL}/search/repositories",
                params={
                    "q": f"{query} mcp-server in:name,description",
                    "sort": "stars",
                    "per_page": limit,
                },
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for repo in data.get("items", []):
                results.append({
                    "name": repo["name"],
                    "description": repo.get("description", ""),
                    "url": repo["html_url"],
                    "stars": repo.get("stargazers_count", 0),
                    "source": "github",
                    "install_command": self._detect_install_command(repo),
                })
            return results

    async def _search_npm(self, query: str, limit: int) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://registry.npmjs.org/-/v1/search",
                params={"text": f"{query} mcp-server", "size": limit},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for obj in data.get("objects", []):
                pkg = obj.get("package", {})
                results.append({
                    "name": pkg.get("name", ""),
                    "description": pkg.get("description", ""),
                    "url": f"https://www.npmjs.com/package/{pkg.get('name', '')}",
                    "version": pkg.get("version", ""),
                    "source": "npm",
                    "install_command": f"npx -y {pkg.get('name', '')}",
                })
            return results

    def _detect_install_command(self, repo: dict[str, Any]) -> str:
        name = repo.get("name", "")
        full_name = repo.get("full_name", "")
        language = repo.get("language", "")

        if language == "Python":
            return f"pip install {name}"
        elif language in ("TypeScript", "JavaScript"):
            return f"npx -y {name}"
        else:
            return f"git clone https://github.com/{full_name}.git"

    async def get_server_info(self, name: str) -> dict[str, Any] | None:
        if name in self._cache:
            return self._cache[name]

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    f"https://registry.npmjs.org/{name}",
                )
                if response.status_code == 200:
                    data = response.json()
                    info = {
                        "name": name,
                        "description": data.get("description", ""),
                        "version": data.get("dist-tags", {}).get("latest", ""),
                        "source": "npm",
                        "install_command": f"npx -y {name}",
                    }
                    self._cache[name] = info
                    return info
        except Exception as e:
            logger.warning("npm_info_failed", name=name, error=str(e))

        return None

    @staticmethod
    def parse_mcp_server_name(name: str) -> dict[str, str]:
        if "/" in name:
            parts = name.split("/")
            return {"scope": parts[0], "name": parts[1]}
        return {"scope": "", "name": name}

    @staticmethod
    def generate_config(
        name: str,
        install_command: str,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        if install_command.startswith("npx"):
            return {
                "name": name,
                "transport": "stdio",
                "command": "npx",
                "args": args or ["-y", name],
                "enabled": True,
            }
        elif install_command.startswith("uvx") or install_command.startswith("pip"):
            return {
                "name": name,
                "transport": "stdio",
                "command": "python",
                "args": args or ["-m", name.replace("-", "_")],
                "enabled": True,
            }
        else:
            return {
                "name": name,
                "transport": "stdio",
                "command": "python",
                "args": args or [f"mcp_servers/{name}.py"],
                "enabled": True,
            }
