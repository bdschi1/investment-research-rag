"""Chunker for Excel financial models.

Ported from Projects/chunkers/excel_models.py. Extracts per-sheet
metadata and previews, serialized as markdown tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.chunking.base import BaseChunker
from rag.chunking.research_chunker import _count_tokens
from rag.chunking.schemas import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class ExcelChunker(BaseChunker):
    """Chunker for Excel financial models — one chunk per sheet."""

    def __init__(self, preview_rows: int = 25):
        self.preview_rows = preview_rows

    def chunk(self, text: str, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        """Chunk from pre-loaded text (markdown-formatted sheets).

        For Excel files, the DocumentLoader returns markdown-formatted
        sheet previews separated by ``---``. This chunker splits on those
        boundaries.
        """
        meta = metadata or ChunkMetadata()
        sheets = text.split("\n\n---\n\n")
        chunks: list[Chunk] = []

        for i, sheet_text in enumerate(sheets):
            sheet_text = sheet_text.strip()
            if not sheet_text:
                continue
            tokens = _count_tokens(sheet_text)
            sheet_meta = ChunkMetadata(
                doc_type=meta.doc_type,
                ticker=meta.ticker,
                filing_date=meta.filing_date,
                section_name=f"sheet_{i}",
                source_filename=meta.source_filename,
            )
            chunks.append(Chunk(
                text=sheet_text,
                metadata=sheet_meta,
                chunk_index=i,
                total_chunks=len(sheets),
                token_count=tokens,
            ))

        logger.info("ExcelChunker produced %d chunks", len(chunks))
        return chunks

    def chunk_file(self, path: str | Path, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        """Chunk directly from an Excel file path.

        Alternative to the text-based ``chunk()`` when you have the raw
        file and want richer metadata per sheet.
        """
        import pandas as pd

        meta = metadata or ChunkMetadata()
        path = Path(path)
        chunks: list[Chunk] = []

        xls = pd.ExcelFile(path)
        total_sheets = len(xls.sheet_names)

        for i, sheet_name in enumerate(xls.sheet_names):
            df = pd.read_excel(xls, sheet_name=sheet_name)
            header = (
                f"## Sheet: {sheet_name} "
                f"({len(df)} rows, {len(df.columns)} cols)\n\n"
            )
            md_table = df.head(self.preview_rows).to_markdown(index=False)
            text = header + (md_table or "[empty sheet]")
            tokens = _count_tokens(text)

            sheet_meta = ChunkMetadata(
                doc_type=meta.doc_type,
                ticker=meta.ticker,
                filing_date=meta.filing_date,
                section_name=sheet_name,
                source_filename=str(path),
            )
            chunks.append(Chunk(
                text=text,
                metadata=sheet_meta,
                chunk_index=i,
                total_chunks=total_sheets,
                token_count=tokens,
            ))

        logger.info("ExcelChunker (file) produced %d chunks from %s", len(chunks), path.name)
        return chunks
