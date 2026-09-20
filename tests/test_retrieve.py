from unittest.mock import MagicMock

from rag_qa.retrieve import retrieve


def test_retrieve_embeds_query_and_delegates_to_store():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    vector_store = MagicMock()
    vector_store.query.return_value = ["chunk1", "chunk2"]

    results = retrieve("a question", embedder, vector_store, top_k=2)

    embedder.embed_query.assert_called_once_with("a question")
    vector_store.query.assert_called_once_with([0.1, 0.2], top_k=2)
    assert results == ["chunk1", "chunk2"]
