"""LLM 统一客户端 — 屏蔽 OpenAI / Anthropic / LiteLLM 差异"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog

from .token_counter import TokenCounter

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: float = 0.0
    finish_reason: str = "stop"


@dataclass
class LLMConfig:
    primary_model: str = "anthropic/mimo-v2.5"
    fallback_model: str = "anthropic/mimo-v2.5"
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 30
    max_retries: int = 3


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.token_counter = TokenCounter()
        self._clients: dict[str, object] = {}

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        target_model = model or self.config.primary_model
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        for attempt in range(self.config.max_retries):
            try:
                start = time.monotonic()
                response = await self._call_model(target_model, messages, temp, tokens)
                latency = (time.monotonic() - start) * 1000

                self.token_counter.record(
                    model=target_model,
                    input_tokens=response.usage.get("prompt_tokens", 0),
                    output_tokens=response.usage.get("completion_tokens", 0),
                )

                logger.info(
                    "llm_call_success",
                    model=target_model,
                    latency_ms=round(latency, 2),
                    tokens=response.usage,
                )
                return response

            except Exception as e:
                logger.warning(
                    "llm_call_failed",
                    model=target_model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if (
                    attempt == self.config.max_retries - 1
                    and target_model != self.config.fallback_model
                ):
                    logger.info(
                        "llm_fallback",
                        from_model=target_model,
                        to_model=self.config.fallback_model,
                    )
                    target_model = self.config.fallback_model
                    attempt = -1

        raise RuntimeError(f"All LLM calls failed after {self.config.max_retries} retries")

    async def _call_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import litellm

        kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.config.timeout,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        response = await litellm.acompletion(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        import litellm

        target_model = model or self.config.primary_model
        kwargs: dict[str, object] = {
            "model": target_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
