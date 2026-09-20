"""Ties the independent stages into the two flows the app needs:

- `build_index`: ingest -> chunk -> embed -> store (offline, run once per corpus change)
- `answer_question`: retrieve -> generate (online, run per query)
"""

import anthropic

from rag_qa.chunk import chunk_documents
from rag_qa.config import Settings, settings
from rag_qa.embed import Embedder
from rag_qa.generate import generate_answer
from rag_qa.ingest import load_documents
from rag_qa.models import Answer
from rag_qa.retrieve import retrieve
from rag_qa.vectorstore import VectorStore


def build_index(config: Settings = settings) -> int:
    """Run the offline ingestion pipeline and populate the vector store.

    Returns:
        The number of chunks indexed.
    """
    documents = load_documents(config.source_dir)
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)

    embedder = Embedder(config.embedding_model)
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = VectorStore(config.chroma_persist_dir, config.collection_name)
    store.reset()
    store.add_chunks(chunks, embeddings)
    return len(chunks)


def answer_question(question: str, config: Settings = settings) -> Answer:
    """Run the online query pipeline: retrieve relevant chunks, then generate."""
    embedder = Embedder(config.embedding_model)
    store = VectorStore(config.chroma_persist_dir, config.collection_name)
    chunks = retrieve(question, embedder, store, top_k=config.top_k)

    client = anthropic.Anthropic()
    return generate_answer(question, chunks, client, model=config.claude_model)
