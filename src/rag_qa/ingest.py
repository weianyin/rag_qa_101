"""Stage 1: load raw documents from disk.

Kept deliberately dumb — plain-text reads, no parsing magic — so failures
in later stages (chunking, embedding) can't be blamed on ingestion.
"""

from pathlib import Path

from rag_qa.models import Document

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(source_dir: str | Path) -> list[Document]:
    """Load every supported file under `source_dir` into a `Document`.

    Args:
        source_dir: Directory to walk recursively for source files.

    Returns:
        One `Document` per file, in sorted path order. `doc_id` is the
        file's path relative to `source_dir`, so it stays stable across runs.

    Raises:
        FileNotFoundError: If `source_dir` does not exist.
    """
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {root}")

    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                doc_id=str(path.relative_to(root)),
                source_path=str(path),
                text=text,
            )
        )
    return documents
