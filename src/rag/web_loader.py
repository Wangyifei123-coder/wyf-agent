"""网页内容加载器 — 提取网页文字和图片"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import structlog
import trafilatura
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@dataclass
class WebContent:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_text_from_html(html: str, url: str) -> str:
    """Extract main text content using trafilatura, removing navigation and ads"""
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
    return text or ""


def extract_images_from_html(html: str, url: str) -> list[str]:
    """Extract all image URLs and convert to absolute paths"""
    soup = BeautifulSoup(html, "html.parser")
    images = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue

        if src.startswith("data:"):
            continue

        abs_url = urljoin(url, src)

        ext = Path(urlparse(abs_url).path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(abs_url)

    seen = set()
    unique = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique.append(img)

    return unique


def extract_title_from_html(html: str) -> str:
    """提取网页标题"""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "Untitled"


async def fetch_webpage(url: str) -> tuple[str, str]:
    """Get webpage HTML content, using Selenium for anti-bot sites"""
    from .selenium_loader import fetch_with_selenium

    try:
        html, final_url = await fetch_with_selenium(url)
        return html, final_url
    except Exception as e:
        logger.warning("selenium_failed", url=url[:80], error=str(e))

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text, str(response.url)


async def download_image(url: str, save_dir: str) -> str | None:
    """Download image to temp directory, return local path"""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            ext = Path(urlparse(url).path).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                ext = ".jpg"

            filename = f"img_{hash(url) % 10**8}{ext}"
            filepath = Path(save_dir) / filename
            filepath.write_bytes(response.content)

            logger.info("image_downloaded", url=url[:80], path=str(filepath))
            return str(filepath)
    except Exception as e:
        logger.warning("image_download_failed", url=url[:80], error=str(e))
        return None


async def load_webpage(url: str) -> WebContent:
    """Load webpage and extract text and images"""
    html, final_url = await fetch_webpage(url)

    title = extract_title_from_html(html)
    text = extract_text_from_html(html, final_url)
    image_urls = extract_images_from_html(html, final_url)

    logger.info(
        "webpage_loaded",
        url=final_url[:80],
        title=title[:50],
        text_len=len(text),
        images=len(image_urls),
    )

    return WebContent(
        url=final_url,
        title=title,
        text=text,
        image_urls=image_urls,
        metadata={"source": final_url, "title": title},
    )
