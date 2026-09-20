"""CLI entrypoint to (re)build the vector index from data/raw.

Usage:
    uv run python scripts/build_index.py
"""

from dotenv import load_dotenv

from rag_qa.config import settings
from rag_qa.pipeline import build_index


def main() -> None:
    load_dotenv()
    num_chunks = build_index(settings)
    print(f"Indexed {num_chunks} chunks from {settings.source_dir} into "
          f"'{settings.collection_name}' at {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
