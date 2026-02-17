"""Abstract base class for all chunkers.

Ported from Projects/chunkers/base_chunker.py and extended with
metadata support and Pydantic-style output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.chunking.schemas import Chunk, ChunkMetadata


class BaseChunker(ABC):
    """Interface for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: ChunkMetadata | None = None) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Full document text.
            metadata: Optional metadata to propagate to each chunk.

        Returns:
            List of ``Chunk`` objects.
        """

    @classmethod
    def strategy_name(cls) -> str:
        """Return human-readable strategy name."""
        return cls.__name__
