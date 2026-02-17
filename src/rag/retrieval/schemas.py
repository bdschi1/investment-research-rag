"""Data models for retrieval operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.vectorstore.schemas import MetadataFilter, SearchResult


@dataclass
class RetrievalConfig:
    """Configuration for a retrieval operation."""

    top_k: int = 10
    rerank: bool = False
    rerank_top_k: int = 5
    metadata_filter: MetadataFilter | None = None
    min_score: float = 0.0


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_candidates: int = 0
    reranked: bool = False
