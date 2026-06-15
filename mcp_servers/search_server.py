"""网页搜索 MCP 服务器 — 提供搜索和网页抓取工具"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WYF Search Server", json_response=True)


@mcp.tool()
async def web_search(query: str, num_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网页

    Args:
        query: 搜索关键词
        num_results: 返回结果数量, 默认 5
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result__body")[:num_results]:
            title_tag = item.select_one(".result__a")
            snippet_tag = item.select_one(".result__snippet")
            url_tag = item.select_one(".result__url")

            title = title_tag.get_text(strip=True) if title_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            url = url_tag.get_text(strip=True) if url_tag else ""

            if title:
                results.append(f"**{title}**\n  URL: {url}\n  {snippet}")

        if not results:
            return f"未找到 '{query}' 的搜索结果"
        return f"搜索 '{query}' 的结果:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"搜索错误: {e}"


@mcp.tool()
async def fetch_url(url: str, max_length: int = 5000) -> str:
    """抓取网页内容并提取正文文本

    Args:
        url: 网页 URL
        max_length: 返回文本最大长度, 默认 5000 字符
    """
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        if len(text) > max_length:
            text = text[:max_length] + "\n...(内容已截断)"

        return f"网页内容 ({url}):\n\n{text}"
    except Exception as e:
        return f"抓取错误: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
