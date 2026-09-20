"""Stage 2: split documents into overlapping, fixed-size chunks.

Fixed-size + overlap is the Week 1 baseline (see README). Week 2 compares
this against other strategies (e.g. sentence/paragraph-aware splitting) with
real retrieval metrics before picking a winner.

Chunk size/overlap are measured in tokens (via tiktoken) rather than
characters so `chunk_size` roughly tracks what actually fits in a prompt,
independent of language/markup density. tiktoken is used purely as a local
tokenizer here — no API calls, no dependency on which model it was built for.
"""

import tiktoken

from rag_qa.models import Chunk, Document

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Split `text` into overlapping chunks of up to `chunk_size` tokens.

    Args:
        text: The text to split.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Tokens shared between consecutive chunks.

    Returns:
        List of (chunk_text, start_char, end_char) tuples, in order.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        token_slice = tokens[start : start + chunk_size]
        chunk_str = _ENCODING.decode(token_slice)

        # Recover character offsets by locating the decoded chunk in the
        # source text; token boundaries don't map 1:1 to char offsets.
        start_char = text.find(chunk_str.strip()[:50]) if chunk_str.strip() else -1
        if start_char == -1:
            start_char = 0
        end_char = start_char + len(chunk_str)

        chunks.append((chunk_str, start_char, end_char))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every document, producing globally unique, ID'd chunks."""
    chunks: list[Chunk] = []
    for doc in documents:
        for i, (text, start_char, end_char) in enumerate(
            chunk_text(doc.text, chunk_size, chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk-{i}",
                    doc_id=doc.doc_id,
                    source_path=doc.source_path,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
    return chunks
