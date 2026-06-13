"""WYF Agent API — FastAPI 应用入口"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .gateway.client import LLMClient, LLMConfig
from .memory.manager import MemoryManager
from .observability.logger import setup_logging
from .observability.tracer import Tracer
from .reasoning.react import ReActEngine
from .safety.guard import SafetyGuard
from .tools.registry import ToolRegistry

_env_path = Path(__file__).parent.parent / "config" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = structlog.get_logger(__name__)

llm_client: LLMClient | None = None
react_engine: ReActEngine | None = None
memory_manager: MemoryManager | None = None
safety_guard: SafetyGuard | None = None
tracer: Tracer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global llm_client, react_engine, memory_manager, safety_guard, tracer

    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), format="json")

    config = LLMConfig(
        primary_model=os.getenv("LLM_PRIMARY_MODEL", "anthropic/mimo-v2.5-pro"),
        fallback_model=os.getenv("LLM_FALLBACK_MODEL", "anthropic/mimo-v2.5-pro"),
        api_base=os.getenv("ANTHROPIC_API_BASE", ""),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )
    llm_client = LLMClient(config)
    memory_manager = MemoryManager()
    safety_guard = SafetyGuard()
    tracer = Tracer()
    tool_registry = ToolRegistry()
    react_engine = ReActEngine(llm_client, tool_registry, memory_manager)

    logger.info("agent_initialized")
    yield
    logger.info("agent_shutdown")


app = FastAPI(title="WYF Agent", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    steps: list[dict[str, Any]] | None = None
    tokens_used: int = 0
    model: str = ""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    assert safety_guard and react_engine and memory_manager

    safety_check = safety_guard.check_input(request.message)
    if not safety_check.safe:
        return ChatResponse(answer=f"Input rejected: {safety_check.reason}")

    result = await react_engine.run(request.message)

    output_check = safety_guard.check_output(result.answer)
    if not output_check.safe:
        result.answer = safety_guard.redact_pii(result.answer)

    return ChatResponse(
        answer=result.answer,
        steps=[{"type": s.type.value, "content": s.content} for s in result.steps],
        tokens_used=result.total_tokens,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    assert llm_client and tracer
    return {
        "token_usage": llm_client.token_counter.summary(),
        "tracing": tracer.summary(),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
