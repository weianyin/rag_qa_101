from rag_qa.models import Chunk
from rag_qa.vectorstore import VectorStore


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc.txt",
        source_path="/doc.txt",
        text=f"text for {chunk_id}",
        start_char=0,
        end_char=10,
    )


def test_add_and_query_roundtrip(tmp_path):
    store = VectorStore(str(tmp_path), "test_collection")
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    store.add_chunks(chunks, embeddings)

    assert store.count() == 2
    results = store.query([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_reset_clears_collection(tmp_path):
    store = VectorStore(str(tmp_path), "test_collection")
    store.add_chunks([_make_chunk("c1")], [[1.0, 0.0]])
    assert store.count() == 1

    store.reset()

    assert store.count() == 0


def test_query_empty_store_returns_nothing(tmp_path):
    store = VectorStore(str(tmp_path), "test_collection")
    assert store.query([1.0, 0.0], top_k=5) == []
