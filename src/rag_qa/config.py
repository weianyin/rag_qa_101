"""Central configuration for the RAG pipeline.

All tunables (chunking, retrieval, model choices) live here so that
experiments in Week 2 can vary them without touching pipeline code.
Values can be overridden via environment variables / a `.env` file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Ingestion
    source_dir: str = "data/raw"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Embeddings — local sentence-transformers model, no external API
    # dependency for the retrieval half of the pipeline.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_qa_101"

    # Retrieval
    top_k: int = 4

    # Generation — Claude API, requires ANTHROPIC_API_KEY in the environment
    claude_model: str = "claude-sonnet-5"
    max_generation_tokens: int = 1024


settings = Settings()
