"""记忆管理器 — 三级记忆协调"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    def __init__(self, max_messages: int = 50, summary_threshold: int = 40) -> None:
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.summary: str = ""

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._compress()

    def get_context(self, last_n: int = 20) -> list[dict[str, str]]:
        recent = self.messages[-last_n:]
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"Previous conversation summary: {self.summary}"})
        for msg in recent:
            context.append({"role": msg.role, "content": msg.content})
        return context

    def _compress(self) -> None:
        overflow = self.messages[: len(self.messages) - self.summary_threshold]
        new_summary_parts = [m.content[:200] for m in overflow[-5:]]
        self.summary = (self.summary + " | " + " ".join(new_summary_parts)).strip(" | ")
        self.messages = self.messages[len(overflow) :]
        logger.info("memory_compressed", remaining=len(self.messages), summary_length=len(self.summary))

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""


class LongTermMemory:
    def __init__(self, collection_name: str = "wyf-agent-memory") -> None:
        self.collection_name = collection_name
        self._store: dict[str, dict[str, Any]] = {}

    async def store(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._store[key] = {"content": content, "metadata": metadata or {}}
        logger.info("long_term_store", key=key, content_length=len(content))

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = []
        query_lower = query.lower()
        for key, entry in self._store.items():
            content = entry["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content)
            if score > 0:
                results.append({"key": key, "content": entry["content"], "score": score, **entry["metadata"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class WorkingMemory:
    def __init__(self, max_size: int = 2000) -> None:
        self.scratchpad: dict[str, Any] = {}
        self.max_size = max_size

    def set(self, key: str, value: Any) -> None:
        self.scratchpad[key] = value

    def get(self, key: str) -> Any:
        return self.scratchpad.get(key)

    def get_all(self) -> dict[str, Any]:
        return dict(self.scratchpad)

    def clear(self) -> None:
        self.scratchpad.clear()


class MemoryManager:
    def __init__(
        self,
        max_messages: int = 50,
        summary_threshold: int = 40,
        collection_name: str = "wyf-agent-memory",
    ) -> None:
        self.short_term = ShortTermMemory(max_messages, summary_threshold)
        self.long_term = LongTermMemory(collection_name)
        self.working = WorkingMemory()

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        from datetime import datetime, timezone

        msg = Message(role=role, content=content, timestamp=datetime.now(timezone.utc).isoformat(), metadata=metadata)
        self.short_term.add(msg)

    def get_context(self, last_n: int = 20) -> list[dict[str, str]]:
        return self.short_term.get_context(last_n)

    async def remember(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        await self.long_term.store(key, content, metadata)

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self.long_term.retrieve(query, top_k)

    def set_task_state(self, key: str, value: Any) -> None:
        self.working.set(key, value)

    def get_task_state(self, key: str) -> Any:
        return self.working.get(key)
