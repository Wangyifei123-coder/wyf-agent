"""Selenium-based webpage loader for anti-bot sites"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import structlog

logger = structlog.get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


def _fetch_sync(url: str) -> tuple[str, str]:
    """Fetch webpage using Selenium in sync context"""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.set_page_load_timeout(30)
        driver.get(url)

        import time
        time.sleep(2)

        html = driver.page_source
        final_url = driver.current_url

        logger.info("selenium_fetched", url=url[:80], html_len=len(html))
        return html, final_url
    finally:
        driver.quit()


async def fetch_with_selenium(url: str) -> tuple[str, str]:
    """Fetch webpage using Selenium (async wrapper)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_sync, url)
