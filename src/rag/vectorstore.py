"""ChromaDB vector store wrapper for RAG pipeline."""

from __future__ import annotations

import chromadb
import structlog

from src.rag.embeddings import EmbeddingService
from src.rag.loader import Document

logger = structlog.get_logger(__name__)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "wyf-agent-kb",
        embedding_service: EmbeddingService | None = None,
        persist_directory: str = "data/chroma",
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vectorstore_initialized",
            collection=collection_name,
            persist_directory=persist_directory,
        )

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        texts = [doc.content for doc in docs]
        embeddings = self.embedding_service.embed_texts(texts)
        ids = [f"doc_{i}_{hash(doc.content) % 10**8}" for i, doc in enumerate(docs)]
        metadatas = [doc.metadata for doc in docs]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        logger.info("documents_added", count=len(docs))

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        query_embedding = self.embedding_service.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents: list[Document] = []
        if results["documents"] and results["documents"][0]:
            for text, metadata in zip(
                results["documents"][0],
                results["metadatas"][0],  # type: ignore[index]
                strict=False,
            ):
                documents.append(Document(content=text, metadata=metadata))
        return documents

    def delete(self, where: dict[str, str]) -> None:
        self._collection.delete(where=where)  # type: ignore[arg-type]
        logger.info("documents_deleted", where=where)

    def delete_by_source(self, source: str) -> int:
        try:
            results = self._collection.get(
                where={"source": source},
            )
            if results and results["ids"]:
                count = len(results["ids"])
                self._collection.delete(ids=results["ids"])
                logger.info("deleted_by_source", source=source, count=count)
                return count
        except Exception as e:
            logger.warning("delete_by_source_failed", source=source, error=str(e))
        return 0

    def get_all_sources(self) -> list[str]:
        try:
            results = self._collection.get(include=["metadatas"])
            if results and results["metadatas"]:
                sources = set()
                for meta in results["metadatas"]:
                    if isinstance(meta, dict) and "source" in meta:
                        sources.add(meta["source"])
                return list(sources)
        except Exception:
            pass
        return []

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore_cleared")

    def get_collection_stats(self) -> dict[str, int | str]:
        return {
            "count": self._collection.count(),
            "name": self._collection.name,
        }
