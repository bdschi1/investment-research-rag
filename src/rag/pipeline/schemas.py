"""Data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """A source citation in a generated answer."""

    index: int
    text: str
    source: str
    section: str | None = None
    ticker: str | None = None
    score: float = 0.0


@dataclass
class RAGQuery:
    """Input to the RAG pipeline."""

    question: str
    ticker: str | None = None
    doc_type: str | None = None
    top_k: int = 10
    rerank: bool = False


@dataclass
class RAGResponse:
    """Output of the RAG pipeline."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    context_texts: list[str] = field(default_factory=list)
    model: str = ""
    retrieval_count: int = 0


@dataclass
class IngestResult:
    """Result of document ingestion."""

    source: str
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    warnings: list[str] = field(default_factory=list)
