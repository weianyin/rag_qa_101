"""Stage 5: embed a query and fetch the top-k matching chunks."""

from rag_qa.embed import Embedder
from rag_qa.models import RetrievedChunk
from rag_qa.vectorstore import VectorStore


def retrieve(
    question: str,
    embedder: Embedder,
    vector_store: VectorStore,
    top_k: int,
) -> list[RetrievedChunk]:
    """Embed `question` and return the `top_k` most similar chunks."""
    query_embedding = embedder.embed_query(question)
    return vector_store.query(query_embedding, top_k=top_k)
