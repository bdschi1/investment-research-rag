"""Token-aware chunker for equity research reports.

Ported from Projects/chunkers/equity_research.py. Splits on paragraph
boundaries, respects section breaks (Exhibit/Figure/Table), and removes
disclosure appendices.
"""

from __future__ import annotations

import logging
import re

from rag.chunking.base import BaseChunker
from rag.chunking.schemas import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

MAX_TOKENS = 800

# Disclosure patterns — stop chunking when these appear
_DISCARD_PATTERNS = [
    re.compile(r"(?i)disclosure\s+appendix"),
    re.compile(r"(?i)important\s+disclosures?"),
    re.compile(r"(?i)regulatory\s+disclosures?"),
    re.compile(r"(?i)analyst\s+certifications?"),
    re.compile(r"(?i)not\s+an?\s+offer\s+to\s+sell"),
    re.compile(r"(?i)past\s+performance"),
]

# New section boundary (exhibits, figures, tables)
_SECTION_BREAK = re.compile(r"^(?:exhibit|figure|table)\s+\d+", re.IGNORECASE)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with graceful fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough approximation: ~1.33 tokens per word
        return len(text.split()) * 4 // 3


class ResearchChunker(BaseChunker):
    """Chunker for equity research reports."""

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens

    def chunk(self, text: str, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        meta = metadata or ChunkMetadata()

        # Truncate at first disclosure pattern
        text = self._truncate_at_disclosures(text)

        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Section break forces a new chunk
            if _SECTION_BREAK.match(para) and current:
                chunks.append(self._make_chunk("\n\n".join(current), current_tokens, meta))
                current = []
                current_tokens = 0

            para_tokens = _count_tokens(para)

            # Would exceed budget — save current and start new
            if current and current_tokens + para_tokens > self.max_tokens:
                chunks.append(self._make_chunk("\n\n".join(current), current_tokens, meta))
                current = []
                current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        # Final chunk
        if current:
            chunks.append(self._make_chunk("\n\n".join(current), current_tokens, meta))

        # Number the chunks
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.chunk_index = i
            c.total_chunks = total

        logger.info(
            "ResearchChunker produced %d chunks from %d chars",
            len(chunks), len(text),
        )
        return chunks

    @staticmethod
    def _truncate_at_disclosures(text: str) -> str:
        lower = text.lower()
        earliest = len(text)
        for pattern in _DISCARD_PATTERNS:
            m = pattern.search(lower)
            if m and m.start() < earliest:
                earliest = m.start()
        return text[:earliest]

    @staticmethod
    def _make_chunk(text: str, token_count: int, meta: ChunkMetadata) -> Chunk:
        return Chunk(text=text, metadata=meta, token_count=token_count)
