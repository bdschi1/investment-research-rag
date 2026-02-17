"""Data models for vector store operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.chunking.schemas import ChunkMetadata


@dataclass
class VectorRecord:
    """A document chunk with its embedding, ready for storage."""

    id: str
    text: str
    embedding: list[float]
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


@dataclass(frozen=True)
class SearchResult:
    """A single search result from the vector store."""

    id: str
    text: str
    score: float
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


@dataclass
class MetadataFilter:
    """Filter search results by metadata fields.

    All specified fields must match (AND logic).
    """

    ticker: str | None = None
    doc_type: str | None = None
    section_name: str | None = None
    item_number: str | None = None
    speaker: str | None = None
    source_filename: str | None = None

    def matches(self, meta: ChunkMetadata) -> bool:
        """Check if a chunk's metadata matches this filter."""
        if self.ticker and meta.ticker != self.ticker:
            return False
        if self.doc_type and meta.doc_type.value != self.doc_type:
            return False
        if self.section_name and meta.section_name != self.section_name:
            return False
        if self.item_number and meta.item_number != self.item_number:
            return False
        if self.speaker and meta.speaker != self.speaker:
            return False
        return not (
            self.source_filename
            and meta.source_filename != self.source_filename
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for Qdrant-style filtering."""
        d: dict[str, Any] = {}
        if self.ticker:
            d["ticker"] = self.ticker
        if self.doc_type:
            d["doc_type"] = self.doc_type
        if self.section_name:
            d["section_name"] = self.section_name
        if self.item_number:
            d["item_number"] = self.item_number
        if self.speaker:
            d["speaker"] = self.speaker
        if self.source_filename:
            d["source_filename"] = self.source_filename
        return d
