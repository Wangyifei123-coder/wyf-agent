"""WYF Agent API — FastAPI 应用入口"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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
from .rag.bm25_retriever import BM25Retriever
from .rag.embeddings import EmbeddingService
from .rag.graph import RAGGraph
from .rag.hybrid_retriever import HybridRetriever
from .rag.kg_retriever import KnowledgeGraphRetriever
from .rag.loader import load_directory
from .rag.retriever import Retriever
from .rag.vectorstore import VectorStore
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
rag_graph: RAGGraph | None = None
vector_store: VectorStore | None = None
hybrid_retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global llm_client, react_engine, memory_manager, safety_guard
    global tracer, rag_graph, vector_store, hybrid_retriever

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

    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)
    vector_retriever = Retriever(vector_store)

    bm25_retriever = BM25Retriever()

    kg_retriever = KnowledgeGraphRetriever()

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        kg_retriever=kg_retriever,
    )

    rag_graph = RAGGraph(llm_client, hybrid_retriever)

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
    assert safety_guard and rag_graph and memory_manager

    safety_check = safety_guard.check_input(request.message)
    if not safety_check.safe:
        return ChatResponse(answer=f"Input rejected: {safety_check.reason}")

    result = await rag_graph.run(request.message)
    answer = result.get("answer", "")

    output_check = safety_guard.check_output(answer)
    if not output_check.safe:
        answer = safety_guard.redact_pii(answer)

    memory_manager.add_message("user", request.message)
    memory_manager.add_message("assistant", answer)

    return ChatResponse(
        answer=answer,
        steps=[{"type": "answer", "content": answer}],
        tokens_used=0,
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


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int


@app.post("/knowledge/ingest", response_model=IngestResponse)
async def ingest_knowledge(request: IngestRequest) -> IngestResponse:
    assert vector_store and hybrid_retriever
    from .rag.splitter import split_documents

    docs = load_directory(request.path)
    chunks = split_documents(docs)
    vector_store.add_documents(chunks)
    hybrid_retriever.index(chunks)
    logger.info("knowledge_ingested", path=request.path, docs=len(docs), chunks=len(chunks))
    return IngestResponse(documents_loaded=len(docs), chunks_created=len(chunks))


@app.get("/knowledge/stats")
async def knowledge_stats() -> dict[str, Any]:
    assert vector_store
    return vector_store.get_collection_stats()


def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
