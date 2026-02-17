"""Speaker-turn chunker for earnings transcripts."""

from __future__ import annotations

import logging

from rag.chunking.base import BaseChunker
from rag.chunking.research_chunker import _count_tokens
from rag.chunking.schemas import Chunk, ChunkMetadata
from rag.documents.transcript_parser import parse_transcript

logger = logging.getLogger(__name__)

MAX_TOKENS = 800


class TranscriptChunker(BaseChunker):
    """Chunker that splits earnings transcripts by speaker turn."""

    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens

    def chunk(self, text: str, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        meta = metadata or ChunkMetadata()
        parsed = parse_transcript(text)

        if not parsed.sections:
            # Fallback: treat whole text as one chunk
            tokens = _count_tokens(text)
            return [Chunk(text=text, metadata=meta, token_count=tokens)]

        chunks: list[Chunk] = []
        for section in parsed.sections:
            section_meta = ChunkMetadata(
                doc_type=meta.doc_type,
                ticker=meta.ticker,
                filing_date=meta.filing_date,
                section_name=section.section_type.value,
                speaker=section.speaker,
                source_filename=meta.source_filename,
            )

            tokens = _count_tokens(section.text)
            if tokens <= self.max_tokens:
                chunks.append(Chunk(
                    text=section.text,
                    metadata=section_meta,
                    token_count=tokens,
                ))
            else:
                # Sub-chunk long speaker turns
                sub = self._split_long_turn(section.text, section_meta)
                chunks.extend(sub)

        total = len(chunks)
        for i, c in enumerate(chunks):
            c.chunk_index = i
            c.total_chunks = total

        logger.info(
            "TranscriptChunker produced %d chunks from %d speaker turns",
            len(chunks), len(parsed.sections),
        )
        return chunks

    def _split_long_turn(self, text: str, meta: ChunkMetadata) -> list[Chunk]:
        """Split a long speaker turn on sentence boundaries."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            if current and current_tokens + sent_tokens > self.max_tokens:
                chunks.append(Chunk(
                    text=" ".join(current),
                    metadata=meta,
                    token_count=current_tokens,
                ))
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(Chunk(
                text=" ".join(current),
                metadata=meta,
                token_count=current_tokens,
            ))
        return chunks
