"""BM25 检索器 — 基于关键词的精确匹配检索"""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from .loader import Document

logger = structlog.get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    current_word: list[str] = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            tokens.append(char)
        elif char.isalnum():
            current_word.append(char)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
    if current_word:
        tokens.append("".join(current_word))
    return tokens


class BM25Retriever:
    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def index(self, documents: list[Document]) -> None:
        self._documents = documents
        self._tokenized_corpus = [_tokenize(doc.content) for doc in documents]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("bm25_indexed", documents=len(documents))

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        if not self._bm25 or not self._documents:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored_docs = list(zip(scores, self._documents, strict=False))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results: list[Document] = []
        for score, doc in scored_docs[:top_k]:
            new_doc = Document(
                content=doc.content,
                metadata={**doc.metadata, "bm25_score": float(score)},
            )
            results.append(new_doc)

        logger.info("bm25_search", query=query[:50], results=len(results))
        return results

    def clear(self) -> None:
        self._documents.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
