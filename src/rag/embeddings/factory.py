"""Embedding provider factory — registry, lazy import, singleton cache.

Follows the same pattern as chunking/factory.py and the data provider
factories in multi-agent-investment-committee.
"""

from __future__ import annotations

import importlib
import logging

from rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry: (provider_key, module_path, class_name)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: list[tuple[str, str, str]] = [
    ("ollama", "rag.embeddings.ollama_provider", "OllamaEmbeddingProvider"),
    ("openai", "rag.embeddings.openai_provider", "OpenAIEmbeddingProvider"),
    ("huggingface", "rag.embeddings.huggingface_provider", "HuggingFaceEmbeddingProvider"),
]

# Singleton cache
_provider_cache: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(
    provider: str = "ollama",
    **kwargs,
) -> EmbeddingProvider:
    """Get an embedding provider by name.

    Args:
        provider: One of ``ollama``, ``openai``, ``huggingface``.
        **kwargs: Passed to the provider constructor.

    Returns:
        An ``EmbeddingProvider`` instance.
    """
    key = provider.lower()

    if not kwargs and key in _provider_cache:
        return _provider_cache[key]

    for reg_key, module_path, cls_name in _PROVIDER_REGISTRY:
        if reg_key == key:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            instance = cls(**kwargs)
            if not kwargs:
                _provider_cache[key] = instance
            return instance

    available = [k for k, _, _ in _PROVIDER_REGISTRY]
    raise ValueError(f"Unknown embedding provider '{provider}'. Available: {available}")


def available_providers() -> list[str]:
    """Return names of registered embedding providers."""
    return [k for k, _, _ in _PROVIDER_REGISTRY]


def clear_cache() -> None:
    """Clear singleton cache (for testing)."""
    _provider_cache.clear()
