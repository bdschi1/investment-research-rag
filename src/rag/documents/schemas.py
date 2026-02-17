"""Data models for document ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DocType(StrEnum):
    """Supported financial document types."""

    SEC_FILING = "sec_filing"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    RESEARCH_REPORT = "research_report"
    FINANCIAL_MODEL = "financial_model"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata attached to a loaded document."""

    doc_type: DocType = DocType.OTHER
    ticker: str | None = None
    filing_date: str | None = None
    filing_type: str | None = None  # e.g. "10-K", "10-Q", "8-K"
    source_url: str | None = None


@dataclass
class LoadResult:
    """Result of loading a single document file.

    Attributes:
        text: Full extracted text.
        page_texts: Per-page text (for PDFs). Empty for non-paged formats.
        source_path: Filesystem path or identifier.
        format: File extension used (pdf, docx, txt, xlsx).
        page_count: Number of pages (PDFs) or sheets (Excel).
        char_count: Length of ``text``.
        metadata: User-supplied or auto-detected metadata.
        warnings: Non-fatal issues encountered during loading.
    """

    text: str
    page_texts: list[str] = field(default_factory=list)
    source_path: str | None = None
    format: str = ""
    page_count: int | None = None
    char_count: int = 0
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    warnings: list[str] = field(default_factory=list)
