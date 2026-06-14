"""检索器 — 向量检索 + RRF 重排序"""
from __future__ import annotations

import structlog

from .loader import Document
from .vectorstore import VectorStore

logger = structlog.get_logger(__name__)


def rrf_score(ranks: list[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        candidates = self.vector_store.search(query, top_k=top_k * 2)
        for i, doc in enumerate(candidates):
            doc.metadata["rrf_rank"] = i + 1
        scored: dict[str, tuple[float, Document]] = {}
        for doc in candidates:
            key = doc.content[:100]
            rank = doc.metadata.get("rrf_rank", 99)
            score = rrf_score([rank])
            if key not in scored or score > scored[key][0]:
                scored[key] = (score, doc)
        sorted_results = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in sorted_results[:top_k]]
        for i, doc in enumerate(results):
            doc.metadata["final_rank"] = i + 1
        logger.info("retrieval_complete", query=query[:50], candidates=len(candidates), results=len(results))
        return results
