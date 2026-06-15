from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import structlog

from src.gateway.client import LLMClient

logger = structlog.get_logger(__name__)

DESCRIBE_PROMPT = (
    "请详细描述这张图片的内容，包括："
    "1. 图片类型（照片、图表、截图等）"
    "2. 主要内容和对象 3. 文字信息（如果有）"
    "4. 其他重要细节"
)

MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


async def describe_image(llm: LLMClient, image_path: str) -> str:
    image_data = Path(image_path).read_bytes()
    image_base64 = base64.b64encode(image_data).decode()

    ext = Path(image_path).suffix.lower().replace(".", "")
    mime_type = MIME_MAP.get(ext, "image/png")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
            ],
        }
    ]

    response = await llm.chat(messages)
    description = response.content.strip()
    logger.info("image_described", path=image_path, description_len=len(description))
    return description
