"""Data models for chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.documents.schemas import DocType


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata carried by each chunk — stored alongside embeddings."""

    doc_type: DocType = DocType.OTHER
    ticker: str | None = None
    filing_date: str | None = None
    section_name: str | None = None
    item_number: str | None = None
    speaker: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    source_filename: str | None = None


@dataclass
class Chunk:
    """A single retrievable piece of a document."""

    text: str
    metadata: ChunkMetadata
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
