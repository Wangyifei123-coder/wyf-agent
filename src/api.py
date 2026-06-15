"""WYF Agent API — FastAPI 应用入口"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from pydantic import BaseModel

from .gateway.client import LLMClient, LLMConfig
from .memory.manager import MemoryManager
from .observability.logger import setup_logging
from .observability.metrics import (
    ACTIVE_SESSIONS,
    CHAT_COUNT,
    KNOWLEDGE_DOCS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from .observability.tracer import Tracer
from .rag.bm25_retriever import BM25Retriever
from .rag.embeddings import EmbeddingService
from .rag.graph import RAGGraph
from .rag.hybrid_retriever import HybridRetriever
from .rag.image_describer import describe_image
from .rag.kg_retriever import KnowledgeGraphRetriever
from .rag.loader import load_directory
from .rag.retriever import Retriever
from .rag.vectorstore import VectorStore
from .reasoning.react import ReActEngine
from .safety.auth import authenticate, get_user_role, verify_token
from .safety.guard import SafetyGuard
from .tools.mcp_manager import MCPManager, MCPServerConfig
from .tools.mcp_registry import MCPRegistryClient
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
mcp_manager: MCPManager | None = None
mcp_registry: MCPRegistryClient | None = None
tool_registry: ToolRegistry | None = None

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U0000fe0f"
    "\U0000200d"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text).strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global llm_client, react_engine, memory_manager, safety_guard
    global tracer, rag_graph, vector_store, hybrid_retriever
    global mcp_manager, mcp_registry, tool_registry

    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), format="json")

    config = LLMConfig(
        primary_model=os.getenv("LLM_PRIMARY_MODEL", "anthropic/mimo-v2.5"),
        fallback_model=os.getenv("LLM_FALLBACK_MODEL", "anthropic/mimo-v2.5"),
        api_base=os.getenv("ANTHROPIC_API_BASE", ""),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        stream_timeout=int(os.getenv("LLM_STREAM_TIMEOUT", "120")),
    )
    llm_client = LLMClient(config)
    memory_manager = MemoryManager()
    safety_guard = SafetyGuard()
    tracer = Tracer()
    tool_registry = ToolRegistry()

    mcp_manager = MCPManager()
    mcp_registry = MCPRegistryClient()
    mcp_config_path = Path(__file__).parent.parent / "config" / "mcp_servers.yaml"
    if mcp_config_path.exists():
        try:
            mcp_configs = MCPManager.load_config(str(mcp_config_path))
            for cfg in mcp_configs:
                await mcp_manager.connect_server(cfg)
            from .tools.mcp_adapter import MCPToolAdapter
            mcp_tools = [
                MCPToolAdapter(t, mcp_manager)
                for t in mcp_manager.get_all_tools()
            ]
            tool_registry.register_mcp_tools(mcp_tools)
            logger.info("mcp_initialized", tools=len(mcp_tools))
        except Exception as e:
            logger.warning("mcp_init_failed", error=str(e))

    embedding_service = EmbeddingService()
    logger.info("prewarming_embedding_model")
    embedding_service.embed_query("warmup")
    logger.info("embedding_model_ready")

    vector_store = VectorStore(embedding_service=embedding_service)
    vector_retriever = Retriever(vector_store)

    bm25_retriever = BM25Retriever()

    kg_retriever = KnowledgeGraphRetriever()

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        kg_retriever=kg_retriever,
    )

    rag_graph = RAGGraph(llm_client, hybrid_retriever, tool_registry)

    react_engine = ReActEngine(llm_client, tool_registry, memory_manager)

    logger.info("agent_initialized")
    yield
    logger.info("agent_shutdown")


app = FastAPI(title="WYF Agent", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    images: list[str] | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    steps: list[dict[str, Any]] | None = None
    tokens_used: int = 0
    model: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    token: str = ""
    username: str = ""
    error: str = ""


def _get_current_user(authorization: str | None = Header(None)) -> str | None:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "")
    result = verify_token(token)
    if result.success:
        return result.username
    return None


def _get_current_user_role(authorization: str | None = Header(None)) -> str:
    if not authorization:
        return "user"
    token = authorization.replace("Bearer ", "")
    return get_user_role(token)


@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    result = authenticate(request.username, request.password)
    return LoginResponse(
        success=result.success,
        token=result.token,
        username=result.username,
        error=result.error,
    )


@app.get("/auth/verify")
async def verify(authorization: str | None = Header(None)) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if user:
        return {"valid": True, "username": user}
    return {"valid": False}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(None),
) -> ChatResponse:
    assert safety_guard and rag_graph and memory_manager

    import time
    start = time.monotonic()

    user = _get_current_user(authorization)
    if not user:
        REQUEST_COUNT.labels(endpoint="/chat", status="unauthorized").inc()
        return ChatResponse(answer="Unauthorized. Please login first.")

    safety_check = safety_guard.check_input(request.message)
    if not safety_check.safe:
        REQUEST_COUNT.labels(endpoint="/chat", status="rejected").inc()
        return ChatResponse(answer=f"Input rejected: {safety_check.reason}")

    memory_context = ""
    try:
        recalled = await memory_manager.recall(request.message, top_k=3)
        if recalled:
            memory_context = "\n".join([r["content"] for r in recalled])
            logger.info("memory_recalled", count=len(recalled))
    except Exception as e:
        logger.warning("memory_recall_failed", error=str(e))

    enhanced_message = request.message
    if memory_context:
        enhanced_message = f"[相关记忆]\n{memory_context}\n\n[用户问题]\n{request.message}"

    result = await rag_graph.run(enhanced_message, images=request.images)
    answer = _strip_emoji(result.get("answer", ""))
    intent = result.get("intent", "knowledge_qa")
    if hasattr(intent, "value"):
        intent = intent.value

    output_check = safety_guard.check_output(answer)
    if not output_check.safe:
        answer = safety_guard.redact_pii(answer)

    memory_manager.add_message("user", request.message)
    memory_manager.add_message("assistant", answer)

    try:
        importance = 0.5
        if any(kw in request.message for kw in ["记住", "记下", "remember", "重要"]):
            importance = 0.9
        elif "?" in request.message or "？" in request.message:
            importance = 0.3

        await memory_manager.remember(
            key=f"conv_{int(time.time())}",
            content=f"用户: {request.message}\n助手: {answer[:200]}",
            importance=importance,
            metadata={"user": user, "intent": str(intent)},
        )
    except Exception as e:
        logger.warning("memory_remember_failed", error=str(e))

    elapsed = time.monotonic() - start
    REQUEST_COUNT.labels(endpoint="/chat", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/chat").observe(elapsed)
    CHAT_COUNT.labels(intent=str(intent)).inc()
    ACTIVE_SESSIONS.inc()

    return ChatResponse(
        answer=answer,
        steps=[{"type": "answer", "content": answer}],
        tokens_used=0,
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authorization: str | None = Header(None),
) -> Any:
    import json
    import time

    from fastapi.responses import StreamingResponse

    assert safety_guard and rag_graph and memory_manager

    user = _get_current_user(authorization)
    if not user:
        async def unauthorized() -> Any:
            yield f"data: {json.dumps({'error': 'Unauthorized'})}\n\n"
        return StreamingResponse(unauthorized(), media_type="text/event-stream")

    safety_check = safety_guard.check_input(request.message)
    if not safety_check.safe:
        async def rejected() -> Any:
            yield f"data: {json.dumps({'error': safety_check.reason})}\n\n"
        return StreamingResponse(rejected(), media_type="text/event-stream")

    start = time.monotonic()

    memory_context = ""
    try:
        recalled = await memory_manager.recall(request.message, top_k=3)
        if recalled:
            memory_context = "\n".join([r["content"] for r in recalled])
            logger.info("memory_recalled", count=len(recalled))
    except Exception as e:
        logger.warning("memory_recall_failed", error=str(e))

    enhanced_message = request.message
    if memory_context:
        enhanced_message = f"[相关记忆]\n{memory_context}\n\n[用户问题]\n{request.message}"

    async def generate() -> Any:
        full_answer = ""
        intent = "knowledge_qa"
        sources: list[str] = []

        async for event in rag_graph.run_stream(enhanced_message, images=request.images):
            event_type = event.get("type")

            if event_type == "intent":
                intent = event.get("intent", intent)
                sources = event.get("sources", [])
                data = json.dumps(
                    {'type': 'intent', 'intent': intent},
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"

            elif event_type == "progress":
                data = json.dumps(
                    {
                        'type': 'progress',
                        'progress': event.get('progress', 0),
                        'message': event.get('message', ''),
                        'tool_name': event.get('tool_name', ''),
                        'stage': event.get('stage', 'pending'),
                    },
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"

            elif event_type == "chunk":
                chunk = event.get("content", "")
                full_answer += chunk
                data = json.dumps(
                    {'type': 'chunk', 'chunk': chunk},
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"

            elif event_type == "done":
                answer = _strip_emoji(full_answer)
                output_check = safety_guard.check_output(answer)
                if not output_check.safe:
                    answer = safety_guard.redact_pii(answer)

                memory_manager.add_message("user", request.message)
                memory_manager.add_message("assistant", answer)

                try:
                    importance = 0.5
                    if any(kw in request.message for kw in ["记住", "记下", "remember", "重要"]):
                        importance = 0.9
                    elif "?" in request.message or "？" in request.message:
                        importance = 0.3

                    await memory_manager.remember(
                        key=f"conv_{int(time.time())}",
                        content=f"用户: {request.message}\n助手: {answer[:200]}",
                        importance=importance,
                        metadata={"user": user, "intent": str(intent)},
                    )
                except Exception as e:
                    logger.warning("memory_remember_failed", error=str(e))

                elapsed = time.monotonic() - start
                REQUEST_COUNT.labels(endpoint="/chat/stream", status="success").inc()
                REQUEST_LATENCY.labels(endpoint="/chat/stream").observe(elapsed)
                CHAT_COUNT.labels(intent=str(intent)).inc()

                data = json.dumps(
                    {'type': 'done', 'intent': intent, 'sources': sources},
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    assert vector_store and hybrid_retriever and llm_client
    from .rag.splitter import split_documents

    docs = load_directory(request.path)

    for doc in docs:
        if doc.metadata.get("file_type") == "image":
            description = await describe_image(llm_client, doc.metadata["source"])
            doc.content = description
            doc.metadata["file_type"] = "image_description"

    chunks = split_documents(docs)

    sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src:
            sources.add(src)

    for source in sources:
        vector_store.delete_by_source(source)

    vector_store.add_documents(chunks)
    hybrid_retriever.index(chunks)
    logger.info(
        "knowledge_ingested",
        path=request.path,
        docs=len(docs),
        chunks=len(chunks),
        sources_updated=len(sources),
    )
    return IngestResponse(documents_loaded=len(docs), chunks_created=len(chunks))


@app.post("/knowledge/rebuild", response_model=IngestResponse)
async def rebuild_knowledge(request: IngestRequest) -> IngestResponse:
    assert vector_store and hybrid_retriever and llm_client
    from .rag.splitter import split_documents

    vector_store.clear()
    logger.info("knowledge_cleared")

    docs = load_directory(request.path)

    for doc in docs:
        if doc.metadata.get("file_type") == "image":
            description = await describe_image(llm_client, doc.metadata["source"])
            doc.content = description
            doc.metadata["file_type"] = "image_description"

    chunks = split_documents(docs)
    vector_store.add_documents(chunks)
    hybrid_retriever.index(chunks)
    logger.info("knowledge_rebuilt", docs=len(docs), chunks=len(chunks))
    return IngestResponse(documents_loaded=len(docs), chunks_created=len(chunks))


@app.get("/knowledge/sources")
async def knowledge_sources() -> dict[str, Any]:
    assert vector_store
    sources = vector_store.get_all_sources()
    return {"sources": sources, "count": len(sources)}


@app.get("/knowledge/stats")
async def knowledge_stats() -> dict[str, Any]:
    assert vector_store
    return vector_store.get_collection_stats()


class IngestURLRequest(BaseModel):
    url: str


class IngestURLResponse(BaseModel):
    title: str
    text_chunks: int
    images_extracted: int
    status: str


@app.post("/knowledge/ingest-url", response_model=IngestURLResponse)
async def ingest_url(request: IngestURLRequest) -> IngestURLResponse:
    assert vector_store and hybrid_retriever and llm_client
    import tempfile

    from .rag.image_describer import describe_image
    from .rag.loader import Document
    from .rag.splitter import split_documents
    from .rag.web_loader import download_image, load_webpage

    web = await load_webpage(request.url)

    docs: list[Document] = []

    if web.text.strip():
        docs.append(Document(
            content=web.text,
            metadata={"source": web.url, "title": web.title, "file_type": "webpage"},
        ))

    images_count = 0
    if web.image_urls:
        with tempfile.TemporaryDirectory() as tmpdir:
            for img_url in web.image_urls[:20]:
                local_path = await download_image(img_url, tmpdir)
                if local_path:
                    description = await describe_image(llm_client, local_path)
                    docs.append(Document(
                        content=description,
                        metadata={
                            "source": web.url,
                            "title": web.title,
                            "file_type": "image_description",
                        },
                    ))
                    images_count += 1

    chunks = split_documents(docs)
    vector_store.add_documents(chunks)
    hybrid_retriever.index(chunks)

    logger.info(
        "url_ingested",
        url=request.url,
        title=web.title,
        text_chunks=len([d for d in docs if d.metadata.get("file_type") == "webpage"]),
        images=images_count,
    )

    return IngestURLResponse(
        title=web.title,
        text_chunks=len([d for d in docs if d.metadata.get("file_type") == "webpage"]),
        images_extracted=images_count,
        status="success",
    )


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: int | str | None = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: dict[str, Any] | None = None
    id: int | str | None = None


@app.post("/mcp")
async def mcp_jsonrpc(
    request: JsonRpcRequest,
    authorization: str | None = Header(None),
) -> JsonRpcResponse:
    if not mcp_manager:
        return JsonRpcResponse(
            id=request.id,
            error={"code": -32603, "message": "MCP not initialized"},
        )

    user = _get_current_user(authorization)
    if not user:
        return JsonRpcResponse(
            id=request.id,
            error={"code": -32600, "message": "Unauthorized"},
        )

    user_role = _get_current_user_role(authorization)
    method = request.method
    params = request.params or {}

    try:
        if method == "tools/list":
            tools = mcp_manager.get_all_tools()
            return JsonRpcResponse(id=request.id, result={"tools": tools})

        elif method == "tools/call":
            server = params.get("server")
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not server or not tool_name:
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32602, "message": "Missing server or tool name"},
                )

            full_tool_name = f"mcp_{server}_{tool_name}"
            if not tool_registry.check_permission(full_tool_name, user_role):
                logger.warning("mcp_permission_denied", tool=full_tool_name, role=user_role)
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32604, "message": f"Permission denied for tool '{tool_name}'"},
                )

            result = await mcp_manager.call_tool(server, tool_name, arguments)
            content = []
            if hasattr(result, "content") and result.content:
                for c in result.content:
                    if hasattr(c, "text"):
                        content.append({"type": "text", "text": c.text})
            return JsonRpcResponse(id=request.id, result={"content": content})

        else:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {method}"},
            )

    except Exception as e:
        return JsonRpcResponse(
            id=request.id,
            error={"code": -32603, "message": str(e)},
        )


@app.get("/mcp/tools")
async def mcp_tools(authorization: str | None = Header(None)) -> dict[str, Any]:
    if not mcp_manager:
        return {"tools": [], "count": 0}

    user_role = _get_current_user_role(authorization)
    all_tools = mcp_manager.get_all_tools()

    filtered_tools = []
    for tool in all_tools:
        tool_key = f"{tool['server']}:{tool['name']}"
        if tool_registry.check_permission(tool_key, user_role):
            filtered_tools.append(tool)

    return {"tools": filtered_tools, "count": len(filtered_tools)}


@app.get("/mcp/servers")
async def mcp_servers() -> dict[str, Any]:
    if not mcp_manager:
        return {"servers": [], "count": 0}
    servers = mcp_manager.get_connected_servers()
    return {"servers": servers, "count": len(servers)}


@app.get("/tools/permissions")
async def check_tool_permissions(
    tool_name: str,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    user_role = _get_current_user_role(authorization)
    has_permission = tool_registry.check_permission(tool_name, user_role)

    return {
        "tool": tool_name,
        "user": user,
        "role": user_role,
        "allowed": has_permission,
    }


@app.get("/tools/list")
async def list_tools_for_user(
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    user_role = _get_current_user_role(authorization)
    tools = tool_registry.list_tools_for_role(user_role)

    return {
        "tools": [{"name": t.name, "description": t.description} for t in tools],
        "count": len(tools),
        "role": user_role,
    }


@app.get("/mcp/search")
async def mcp_search(
    q: str = "",
    limit: int = 10,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    if not mcp_registry:
        return {"results": [], "count": 0}

    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    if not q:
        return {"results": [], "count": 0}

    results = await mcp_registry.search(q, limit)
    return {"results": results, "count": len(results)}


class MCPInstallRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = "npx"
    args: list[str] = []


@app.post("/mcp/install")
async def mcp_install(
    request: MCPInstallRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    if not mcp_manager:
        return {"success": False, "error": "MCP not initialized"}

    user = _get_current_user(authorization)
    if not user:
        return {"success": False, "error": "Unauthorized"}

    config_path = Path(__file__).parent.parent / "config" / "mcp_servers.yaml"

    try:
        config = MCPServerConfig(
            name=request.name,
            transport=request.transport,
            command=request.command,
            args=request.args,
        )

        await mcp_manager.connect_server(config)

        existing_configs = []
        if config_path.exists():
            existing_configs = MCPManager.load_config(str(config_path))

        new_server = {
            "name": request.name,
            "transport": request.transport,
            "command": request.command,
            "args": request.args,
            "enabled": True,
        }

        all_servers = [vars(c) for c in existing_configs] if existing_configs else []
        all_servers.append(new_server)

        MCPManager.save_config(str(config_path), all_servers)

        from .tools.mcp_adapter import MCPToolAdapter
        mcp_tools = [
            MCPToolAdapter(t, mcp_manager)
            for t in mcp_manager.get_all_tools()
            if t.get("server") == request.name
        ]

        logger.info("mcp_server_installed", server=request.name, tools=len(mcp_tools))
        return {
            "success": True,
            "server": request.name,
            "tools_installed": len(mcp_tools),
        }
    except Exception as e:
        logger.error("mcp_install_failed", server=request.name, error=str(e))
        return {"success": False, "error": str(e)}


@app.get("/prometheus")
async def prometheus_metrics() -> Any:
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    if vector_store:
        stats = vector_store.get_collection_stats()
        KNOWLEDGE_DOCS.set(float(stats.get("count", 0)))

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


class MemoryStoreRequest(BaseModel):
    key: str
    content: str
    importance: float = 0.5
    metadata: dict[str, Any] | None = None


class MemoryRecallRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/memory/store")
async def memory_store(
    request: MemoryStoreRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    assert memory_manager
    try:
        await memory_manager.remember(
            key=request.key,
            content=request.content,
            importance=request.importance,
            metadata=request.metadata,
        )
        return {"success": True, "key": request.key}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/memory/recall")
async def memory_recall(
    request: MemoryRecallRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    assert memory_manager
    try:
        results = await memory_manager.recall(request.query, top_k=request.top_k)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/memory/stats")
async def memory_stats(
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    assert memory_manager
    return {
        "short_term": {
            "messages": len(memory_manager.short_term.messages),
            "has_summary": bool(memory_manager.short_term.summary),
        },
        "long_term": {
            "entries": len(memory_manager.long_term._entries),
            "collection": memory_manager.long_term.collection_name,
        },
        "working": {
            "keys": list(memory_manager.working.scratchpad.keys()),
        },
    }


@app.post("/memory/cleanup")
async def memory_cleanup(
    max_entries: int = 1000,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    user = _get_current_user(authorization)
    if not user:
        return {"error": "Unauthorized"}

    assert memory_manager
    try:
        removed = await memory_manager.cleanup_memory(max_entries)
        return {"success": True, "removed": removed}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
