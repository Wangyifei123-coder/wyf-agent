"""HTTP API MCP 服务器 — 提供通用 HTTP 请求工具"""

from __future__ import annotations

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WYF API Server", json_response=True)


@mcp.tool()
async def http_get(
    url: str,
    headers: str = "{}",
    timeout: int = 30,
) -> str:
    """发送 HTTP GET 请求

    Args:
        url: 请求 URL
        headers: JSON 格式的请求头, 如 '{"Authorization": "Bearer xxx"}'
        timeout: 超时秒数, 默认 30
    """
    try:
        header_dict = json.loads(headers)
    except json.JSONDecodeError:
        return f"错误: headers 不是有效的 JSON: {headers}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=header_dict)
            return _format_response(response)
    except Exception as e:
        return f"请求错误: {e}"


@mcp.tool()
async def http_post(
    url: str,
    body: str = "{}",
    headers: str = '{"Content-Type": "application/json"}',
    timeout: int = 30,
) -> str:
    """发送 HTTP POST 请求

    Args:
        url: 请求 URL
        body: 请求体 (JSON 字符串)
        headers: JSON 格式的请求头
        timeout: 超时秒数, 默认 30
    """
    try:
        header_dict = json.loads(headers)
    except json.JSONDecodeError:
        return f"错误: headers 不是有效的 JSON: {headers}"

    try:
        json_body = json.loads(body)
    except json.JSONDecodeError:
        json_body = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if json_body is not None:
                response = await client.post(
                    url, json=json_body, headers=header_dict
                )
            else:
                response = await client.post(
                    url, content=body.encode(), headers=header_dict
                )
            return _format_response(response)
    except Exception as e:
        return f"请求错误: {e}"


def _format_response(response: httpx.Response) -> str:
    status = response.status_code
    content_type = response.headers.get("content-type", "")

    lines = [f"HTTP {status}"]

    try:
        if "json" in content_type:
            data = response.json()
            lines.append(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            text = response.text[:5000]
            if len(response.text) > 5000:
                text += "\n...(内容已截断)"
            lines.append(text)
    except Exception:
        lines.append(response.text[:5000])

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
