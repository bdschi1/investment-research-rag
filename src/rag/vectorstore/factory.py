"""Vector store factory — registry, lazy import, singleton cache."""

from __future__ import annotations

import importlib
import logging

from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store registry: (store_key, module_path, class_name)
# ---------------------------------------------------------------------------

_STORE_REGISTRY: list[tuple[str, str, str]] = [
    ("faiss", "rag.vectorstore.faiss_store", "FAISSStore"),
    ("qdrant", "rag.vectorstore.qdrant_store", "QdrantStore"),
]

# Singleton cache
_store_cache: dict[str, VectorStore] = {}


def get_vector_store(
    provider: str = "faiss",
    **kwargs,
) -> VectorStore:
    """Get a vector store by name.

    Args:
        provider: One of ``faiss``, ``qdrant``.
        **kwargs: Passed to the store constructor.

    Returns:
        A ``VectorStore`` instance.
    """
    key = provider.lower()

    if not kwargs and key in _store_cache:
        return _store_cache[key]

    for reg_key, module_path, cls_name in _STORE_REGISTRY:
        if reg_key == key:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            instance = cls(**kwargs)
            if not kwargs:
                _store_cache[key] = instance
            return instance

    available = [k for k, _, _ in _STORE_REGISTRY]
    raise ValueError(f"Unknown vector store '{provider}'. Available: {available}")


def available_stores() -> list[str]:
    """Return names of registered vector stores."""
    return [k for k, _, _ in _STORE_REGISTRY]


def clear_cache() -> None:
    """Clear singleton cache (for testing)."""
    _store_cache.clear()
