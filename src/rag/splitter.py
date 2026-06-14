"""Structure-aware text splitter for RAG pipeline."""

from __future__ import annotations

from src.rag.loader import Document

SEPARATORS = ["\n\n", "\n", ".", " ", ""]


def _count_tokens(text: str) -> int:
    return len(text) // 3


def _split_by_separators(
    text: str, chunk_size: int, overlap: int, separators: list[str]
) -> list[str]:
    if _count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        if sep == "":
            chunks: list[str] = []
            start = 0
            while start < len(text):
                end = start + chunk_size * 3
                chunks.append(text[start:end])
                start = end - overlap * 3
            return chunks

        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if _count_tokens(candidate) > chunk_size and current:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)

        if len(chunks) > 1:
            if overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for j in range(1, len(chunks)):
                    prev = chunks[j - 1]
                    overlap_chars = overlap * 3
                    tail = prev[-overlap_chars:] if overlap_chars < len(prev) else prev
                    overlapped.append(tail + sep + chunks[j])
                return overlapped
            return chunks

    return [text] if text.strip() else []


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    return _split_by_separators(text, chunk_size, overlap, SEPARATORS)


def split_documents(
    docs: list[Document], chunk_size: int = 512, overlap: int = 64
) -> list[Document]:
    result: list[Document] = []
    for doc in docs:
        chunks = split_text(doc.content, chunk_size, overlap)
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            meta = {**doc.metadata, "chunk_index": idx, "total_chunks": total}
            result.append(Document(content=chunk, metadata=meta))
    return result
