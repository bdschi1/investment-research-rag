"""Section-aware chunker for SEC filings (10-K, 10-Q).

Uses ``sec_parser.parse_sec_filing()`` to detect ITEM boundaries, then
sub-chunks oversized sections on paragraph boundaries.
"""

from __future__ import annotations

import logging
import re

from rag.chunking.base import BaseChunker
from rag.chunking.research_chunker import _count_tokens
from rag.chunking.schemas import Chunk, ChunkMetadata
from rag.documents.sec_parser import Section, parse_sec_filing

logger = logging.getLogger(__name__)

MAX_TOKENS = 800
OVERLAP_TOKENS = 100


class SecChunker(BaseChunker):
    """Chunker for SEC filings with ITEM-level section awareness."""

    def __init__(self, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        meta = metadata or ChunkMetadata()

        # Split on page breaks if available, otherwise treat as single page
        page_texts = text.split("\n\n") if "\n\n" in text else [text]

        # Try section-aware chunking
        filing = parse_sec_filing(page_texts)

        if filing.has_sections:
            chunks = self._chunk_by_sections(text, filing.sections, meta)
        else:
            # Fall back to paragraph-based chunking
            chunks = self._chunk_paragraphs(text, meta)

        total = len(chunks)
        for i, c in enumerate(chunks):
            c.chunk_index = i
            c.total_chunks = total

        logger.info(
            "SecChunker produced %d chunks (section-aware=%s)",
            len(chunks), filing.has_sections,
        )
        return chunks

    def _chunk_by_sections(
        self,
        full_text: str,
        sections: list[Section],
        meta: ChunkMetadata,
    ) -> list[Chunk]:
        """Create chunks from detected SEC sections, sub-chunking large ones."""
        chunks: list[Chunk] = []

        for section in sections:
            section_text = full_text[section.start_char:section.end_char].strip()
            section_tokens = _count_tokens(section_text)

            section_meta = ChunkMetadata(
                doc_type=meta.doc_type,
                ticker=meta.ticker,
                filing_date=meta.filing_date,
                section_name=section.title,
                item_number=section.item_number,
                page_numbers=list(range(section.start_page, section.end_page + 1)),
                source_filename=meta.source_filename,
            )

            if section_tokens <= self.max_tokens:
                chunks.append(Chunk(
                    text=section_text,
                    metadata=section_meta,
                    token_count=section_tokens,
                ))
            else:
                # Sub-chunk on paragraph boundaries
                sub_chunks = self._split_section(section_text, section_meta)
                chunks.extend(sub_chunks)

        return chunks

    def _split_section(self, text: str, meta: ChunkMetadata) -> list[Chunk]:
        """Split a large section into token-bounded chunks."""
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = _count_tokens(para)

            if current and current_tokens + para_tokens > self.max_tokens:
                chunks.append(Chunk(
                    text="\n\n".join(current),
                    metadata=meta,
                    token_count=current_tokens,
                ))
                current = []
                current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(Chunk(
                text="\n\n".join(current),
                metadata=meta,
                token_count=current_tokens,
            ))

        return chunks

    def _chunk_paragraphs(self, text: str, meta: ChunkMetadata) -> list[Chunk]:
        """Fallback: paragraph-based chunking without section awareness."""
        return self._split_section(text, meta)
