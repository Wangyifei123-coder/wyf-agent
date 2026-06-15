"""记忆管理器 — 三级记忆协调"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    key: str
    content: str
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def decay_score(self, current_time: float | None = None) -> float:
        now = current_time or time.time()
        hours_since_access = (now - self.last_accessed) / 3600
        decay_factor = math.exp(-0.1 * hours_since_access)
        access_bonus = min(self.access_count * 0.05, 0.3)
        return (self.importance + access_bonus) * decay_factor


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
            summary_text = f"Previous conversation summary: {self.summary}"
            context.append({"role": "system", "content": summary_text})
        for msg in recent:
            context.append({"role": msg.role, "content": msg.content})
        return context

    def _compress(self) -> None:
        overflow = self.messages[: len(self.messages) - self.summary_threshold]
        new_summary_parts = [m.content[:200] for m in overflow[-5:]]
        combined = self.summary + " | " + " ".join(new_summary_parts)
        self.summary = combined.strip().strip("|").strip()
        self.messages = self.messages[len(overflow) :]
        logger.info(
            "memory_compressed",
            remaining=len(self.messages),
            summary_length=len(self.summary),
        )

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""


class LongTermMemory:
    def __init__(self, collection_name: str = "wyf-agent-memory") -> None:
        self.collection_name = collection_name
        self._entries: dict[str, MemoryEntry] = {}

    async def store(
        self,
        key: str,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        if key in self._entries:
            entry = self._entries[key]
            entry.content = content
            entry.importance = importance
            entry.access_count += 1
            entry.last_accessed = now
            entry.metadata.update(metadata or {})
        else:
            self._entries[key] = MemoryEntry(
                key=key,
                content=content,
                importance=importance,
                access_count=1,
                last_accessed=now,
                created_at=now,
                metadata=metadata or {},
            )
        logger.info("long_term_store", key=key, importance=importance)

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._entries:
            return []

        now = time.time()
        scored_entries: list[tuple[float, MemoryEntry]] = []

        query_lower = query.lower()
        for entry in self._entries.values():
            content_lower = entry.content.lower()
            keyword_score = sum(1 for word in query_lower.split() if word in content_lower)
            if keyword_score > 0:
                decay_score = entry.decay_score(now)
                final_score = keyword_score * decay_score
                scored_entries.append((final_score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored_entries[:top_k]:
            entry.access_count += 1
            entry.last_accessed = now
            results.append({
                "key": entry.key,
                "content": entry.content,
                "score": score,
                "importance": entry.importance,
                "access_count": entry.access_count,
                **entry.metadata,
            })

        return results

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def cleanup(self, max_entries: int = 1000) -> int:
        if len(self._entries) <= max_entries:
            return 0

        now = time.time()
        entries_with_score = [
            (entry.decay_score(now), key)
            for key, entry in self._entries.items()
        ]
        entries_with_score.sort()

        removed = 0
        for _, key in entries_with_score[: len(self._entries) - max_entries]:
            del self._entries[key]
            removed += 1

        logger.info("memory_cleanup", removed=removed)
        return removed


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
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now(UTC).isoformat(),
            metadata=metadata,
        )
        self.short_term.add(msg)

    def get_context(self, last_n: int = 20) -> list[dict[str, str]]:
        return self.short_term.get_context(last_n)

    async def remember(
        self,
        key: str,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.long_term.store(key, content, importance, metadata)

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self.long_term.retrieve(query, top_k)

    async def cleanup_memory(self, max_entries: int = 1000) -> int:
        return await self.long_term.cleanup(max_entries)

    def set_task_state(self, key: str, value: Any) -> None:
        self.working.set(key, value)

    def get_task_state(self, key: str) -> Any:
        return self.working.get(key)
