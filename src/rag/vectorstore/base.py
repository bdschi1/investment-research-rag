"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord


class VectorStore(ABC):
    """Interface for vector store backends."""

    @abstractmethod
    def add(self, records: list[VectorRecord]) -> int:
        """Insert records into the store.

        Args:
            records: Documents with embeddings.

        Returns:
            Number of records successfully inserted.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query vector.
            top_k: Maximum results to return.
            metadata_filter: Optional metadata filter.

        Returns:
            List of ``SearchResult`` sorted by relevance (highest first).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of records in the store."""

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete records by ID.

        Returns:
            Number of records deleted.
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all records."""

    def save(self, path: str) -> None:
        """Persist the store to disk (optional)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support save()")

    def load(self, path: str) -> None:
        """Load the store from disk (optional)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support load()")

    @classmethod
    def store_name(cls) -> str:
        """Return human-readable store name."""
        return cls.__name__
