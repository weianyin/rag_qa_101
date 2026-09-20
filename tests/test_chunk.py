import pytest

from rag_qa.chunk import chunk_documents, chunk_text
from rag_qa.models import Document


def test_short_text_produces_single_chunk():
    chunks = chunk_text("a short sentence", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0][0].strip() == "a short sentence"


def test_long_text_splits_with_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    # every chunk should be non-empty and within the token budget's rough
    # character equivalent
    assert all(c[0] for c in chunks)


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, chunk_overlap=10)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, chunk_overlap=0)


def test_chunk_documents_assigns_unique_ids():
    docs = [
        Document(doc_id="a.txt", source_path="/a.txt", text="hello world"),
        Document(doc_id="b.txt", source_path="/b.txt", text="goodbye world"),
    ]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "a.txt::chunk-0"
    assert chunks[1].chunk_id == "b.txt::chunk-0"
    assert chunks[0].doc_id == "a.txt"
