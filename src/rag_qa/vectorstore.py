"""Stage 4: persist and query chunk embeddings via Chroma.

Chroma runs embedded/local — no external service to stand up — which is
enough for a portfolio-scale corpus. A hosted store (Pinecone/Weaviate) is
only worth the operational overhead past that scale; see README tradeoffs.
"""

import chromadb

from rag_qa.models import Chunk, RetrievedChunk


class VectorStore:
    """Wraps a persistent Chroma collection of chunk embeddings."""

    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their pre-computed embeddings into the store."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "source_path": c.source_path,
                 "start_char": c.start_char, "end_char": c.end_char}
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` chunks closest to `query_embedding`."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata["doc_id"],
                    source_path=metadata["source_path"],
                    text=results["documents"][0][i],
                    distance=results["distances"][0][i],
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all chunks in the collection. Used when rebuilding the index."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
