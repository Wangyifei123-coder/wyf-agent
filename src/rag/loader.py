"""Document loaders for RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from PyPDF2 import PdfReader

logger = structlog.get_logger()

EXTENSION_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".pdf": "pdf",
}


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_markdown(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    logger.info("loaded_markdown", path=path, chars=len(text))
    return Document(content=text, metadata={"file_type": "markdown", "source": path})


def load_text(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    logger.info("loaded_text", path=path, chars=len(text))
    return Document(content=text, metadata={"file_type": "text", "source": path})


def load_pdf(path: str) -> Document:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    logger.info("loaded_pdf", path=path, pages=len(pages), chars=len(text))
    return Document(content=text, metadata={"file_type": "pdf", "source": path})


_LOADERS = {
    "markdown": load_markdown,
    "text": load_text,
    "pdf": load_pdf,
}


def load_directory(path: str) -> list[Document]:
    docs: list[Document] = []
    for root, _dirs, files in os.walk(path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            ftype = EXTENSION_MAP.get(ext)
            if ftype is None:
                logger.debug("skipping_unsupported", file=fname)
                continue
            fpath = os.path.join(root, fname)
            loader = _LOADERS[ftype]
            docs.append(loader(fpath))
    logger.info("loaded_directory", path=path, documents=len(docs))
    return docs
