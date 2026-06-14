"""混合检索器 — 向量 + BM25 + 知识图谱, RRF 融合"""

from __future__ import annotations

import structlog

from .bm25_retriever import BM25Retriever
from .kg_retriever import KnowledgeGraphRetriever
from .loader import Document
from .retriever import Retriever, rrf_score

logger = structlog.get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        kg_retriever: KnowledgeGraphRetriever | None = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        kg_weight: float = 0.5,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.kg_retriever = kg_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.kg_weight = kg_weight

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        vector_docs = self.vector_retriever.retrieve(query, top_k=top_k)
        bm25_docs = self.bm25_retriever.search(query, top_k=top_k)
        kg_docs: list[Document] = []
        if self.kg_retriever:
            kg_docs = self.kg_retriever.search(query, top_k=top_k)

        doc_scores: dict[str, tuple[float, Document]] = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.content[:200]
            weighted_score = self.vector_weight * rrf_score([rank + 1])
            if key in doc_scores:
                old_score, old_doc = doc_scores[key]
                doc_scores[key] = (old_score + weighted_score, old_doc)
            else:
                doc_scores[key] = (weighted_score, doc)

        for rank, doc in enumerate(bm25_docs):
            key = doc.content[:200]
            weighted_score = self.bm25_weight * rrf_score([rank + 1])
            if key in doc_scores:
                old_score, old_doc = doc_scores[key]
                doc_scores[key] = (old_score + weighted_score, old_doc)
            else:
                doc_scores[key] = (weighted_score, doc)

        for rank, doc in enumerate(kg_docs):
            key = doc.content[:200]
            weighted_score = self.kg_weight * rrf_score([rank + 1])
            if key in doc_scores:
                old_score, old_doc = doc_scores[key]
                doc_scores[key] = (old_score + weighted_score, old_doc)
            else:
                doc_scores[key] = (weighted_score, doc)

        sorted_results = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in sorted_results[:top_k]]

        for i, doc in enumerate(results):
            doc.metadata["final_rank"] = i + 1

        logger.info(
            "hybrid_retrieval",
            query=query[:50],
            vector=len(vector_docs),
            bm25=len(bm25_docs),
            kg=len(kg_docs),
            final=len(results),
        )
        return results

    def index(self, documents: list[Document]) -> None:
        self.bm25_retriever.index(documents)
        if self.kg_retriever:
            self.kg_retriever.add_documents(documents)
        logger.info("hybrid_indexed", documents=len(documents))
