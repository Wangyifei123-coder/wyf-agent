"""Tests for RAG document loader."""

import pytest
from pathlib import Path

from src.rag.loader import Document, load_directory, load_markdown, load_pdf, load_text


class TestDocument:
    def test_creation(self):
        doc = Document(content="hello", metadata={"k": "v"})
        assert doc.content == "hello"
        assert doc.metadata == {"k": "v"}

    def test_default_metadata(self):
        doc = Document(content="x")
        assert doc.metadata == {}


class TestLoadMarkdown:
    def test_reads_content(self, tmp_path: Path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nBody text", encoding="utf-8")
        doc = load_markdown(str(f))
        assert doc.content == "# Title\n\nBody text"
        assert doc.metadata["file_type"] == "markdown"
        assert doc.metadata["source"] == str(f)


class TestLoadText:
    def test_reads_content(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("plain text", encoding="utf-8")
        doc = load_text(str(f))
        assert doc.content == "plain text"
        assert doc.metadata["file_type"] == "text"


class TestLoadPdf:
    def test_reads_pdf(self, tmp_path: Path):
        pdf_path = tmp_path / "sample.pdf"
        _create_blank_pdf(pdf_path)
        doc = load_pdf(str(pdf_path))
        assert doc.metadata["file_type"] == "pdf"
        assert doc.metadata["source"] == str(pdf_path)
        assert isinstance(doc.content, str)


class TestLoadDirectory:
    def test_mixed_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("md", encoding="utf-8")
        (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
        _create_blank_pdf(tmp_path / "c.pdf")
        docs = load_directory(str(tmp_path))
        types = {d.metadata["file_type"] for d in docs}
        assert types == {"markdown", "text", "pdf"}

    def test_skips_unsupported(self, tmp_path: Path):
        (tmp_path / "data.md").write_text("ok", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "script.py").write_text("print()", encoding="utf-8")
        docs = load_directory(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "markdown"

    def test_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.md").write_text("root", encoding="utf-8")
        (sub / "deep.txt").write_text("deep", encoding="utf-8")
        docs = load_directory(str(tmp_path))
        assert len(docs) == 2


def _create_blank_pdf(path: Path) -> None:
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)
