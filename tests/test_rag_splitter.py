from src.rag.loader import Document
from src.rag.splitter import split_documents, split_text


def test_short_text_unchanged():
    text = "Hello world"
    chunks = split_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_split():
    text = "word " * 2000
    chunks = split_text(text, chunk_size=512, overlap=64)
    assert len(chunks) > 1


def test_overlap_present():
    text = "sentence one. sentence two. sentence three. sentence four. " * 50
    chunks = split_text(text, chunk_size=64, overlap=16)
    if len(chunks) > 1:
        assert any(
            chunks[i + 1].startswith(chunks[i][-50:])
            or chunks[i][-50:] in chunks[i + 1]
            for i in range(len(chunks) - 1)
        )


def test_preserves_metadata():
    doc = Document(
        content="word " * 2000,
        metadata={"source": "test.md", "file_type": "markdown"},
    )
    result = split_documents([doc], chunk_size=512, overlap=64)
    assert len(result) > 1
    for i, r in enumerate(result):
        assert r.metadata["source"] == "test.md"
        assert r.metadata["file_type"] == "markdown"
        assert r.metadata["chunk_index"] == i
        assert r.metadata["total_chunks"] == len(result)


def test_markdown_structure_aware():
    text = (
        "# Header One\n\n"
        + "Some content under header one. " * 20
        + "\n\n# Header Two\n\n"
        + "Some content under header two. " * 20
    )
    chunks = split_text(text, chunk_size=128, overlap=32)
    assert len(chunks) >= 2


def test_empty_document():
    doc = Document(content="", metadata={"source": "empty.md"})
    result = split_documents([doc], chunk_size=512, overlap=64)
    assert len(result) == 0
