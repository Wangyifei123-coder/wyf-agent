"""MCP 服务器测试 — db_server / search_server / api_server"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── db_server 测试 ──────────────────────────────────────────────


class TestDbServer:
    def _make_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        conn.commit()
        conn.close()

    def test_query_database_select(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._make_db(db_path)

        from mcp_servers.db_server import query_database

        result = query_database("SELECT * FROM users ORDER BY id", db_path=str(db_path))
        assert "Alice" in result
        assert "Bob" in result

    def test_query_database_rejects_non_select(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._make_db(db_path)

        from mcp_servers.db_server import query_database

        result = query_database("DROP TABLE users", db_path=str(db_path))
        assert "只读" in result

    def test_list_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._make_db(db_path)

        from mcp_servers.db_server import list_tables

        result = list_tables(db_path=str(db_path))
        assert "users" in result

    def test_list_tables_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        from mcp_servers.db_server import list_tables

        result = list_tables(db_path=str(db_path))
        assert "没有表" in result

    def test_get_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._make_db(db_path)

        from mcp_servers.db_server import get_schema

        result = get_schema("users", db_path=str(db_path))
        assert "id" in result
        assert "name" in result
        assert "主键" in result

    def test_get_schema_nonexistent_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._make_db(db_path)

        from mcp_servers.db_server import get_schema

        result = get_schema("nonexistent", db_path=str(db_path))
        assert "不存在" in result

    def test_query_database_file_not_found(self) -> None:
        from mcp_servers.db_server import query_database

        result = query_database("SELECT 1", db_path="/nonexistent/db.sqlite")
        assert "不存在" in result


# ── api_server 测试 ──────────────────────────────────────────────


class TestApiServer:
    @pytest.mark.asyncio
    async def test_http_get_success(self) -> None:
        from mcp_servers.api_server import http_get

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"ok": True}

        with patch("mcp_servers.api_server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await http_get("https://example.com/api")
            assert "200" in result
            assert "ok" in result

    @pytest.mark.asyncio
    async def test_http_post_json(self) -> None:
        from mcp_servers.api_server import http_post

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"created": True}

        with patch("mcp_servers.api_server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await http_post(
                "https://example.com/api",
                body='{"name": "test"}',
            )
            assert "201" in result
            assert "created" in result

    @pytest.mark.asyncio
    async def test_http_get_invalid_headers(self) -> None:
        from mcp_servers.api_server import http_get

        result = await http_get("https://example.com", headers="not-json")
        assert "错误" in result


# ── search_server 测试 ──────────────────────────────────────────


class TestSearchServer:
    @pytest.mark.asyncio
    async def test_web_search_no_results(self) -> None:
        from mcp_servers.search_server import web_search

        html = "<html><body><p>无结果</p></body></html>"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("mcp_servers.search_server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await web_search("xyznonexistent123")
            assert "未找到" in result

    @pytest.mark.asyncio
    async def test_fetch_url_success(self) -> None:
        from mcp_servers.search_server import fetch_url

        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <script>var x = 1;</script>
            <p>Hello World</p>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("mcp_servers.search_server.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await fetch_url("https://example.com")
            assert "Hello World" in result
            assert "var x" not in result  # script 标签应被移除
