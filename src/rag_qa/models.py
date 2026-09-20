"""Shared dataclasses passed between pipeline stages."""

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single ingested source document, before chunking."""

    doc_id: str
    source_path: str
    text: str


@dataclass
class Chunk:
    """A chunk of a document, ready to be embedded."""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    start_char: int
    end_char: int


@dataclass
class RetrievedChunk:
    """A chunk returned by the vector store for a query, with its score."""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    distance: float


@dataclass
class Citation:
    source_path: str
    chunk_id: str
    text: str


@dataclass
class Answer:
    """The result of the generation stage."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
