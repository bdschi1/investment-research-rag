"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors (same order as input).
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Some providers use different models/prefixes for queries vs documents.

        Args:
            query: The search query.

        Returns:
            Embedding vector.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""

    @classmethod
    def provider_name(cls) -> str:
        """Return human-readable provider name."""
        return cls.__name__
