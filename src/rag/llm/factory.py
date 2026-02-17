"""LLM provider factory — registry, lazy import, singleton cache."""

from __future__ import annotations

import importlib
import logging

from rag.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry: (provider_key, module_path, class_name)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: list[tuple[str, str, str]] = [
    ("ollama", "rag.llm.ollama_provider", "OllamaLLMProvider"),
    ("anthropic", "rag.llm.anthropic_provider", "AnthropicLLMProvider"),
    ("openai", "rag.llm.openai_provider", "OpenAILLMProvider"),
]

# Singleton cache
_provider_cache: dict[str, LLMProvider] = {}


def get_llm_provider(
    provider: str = "ollama",
    **kwargs,
) -> LLMProvider:
    """Get an LLM provider by name.

    Args:
        provider: One of ``ollama``, ``anthropic``, ``openai``.
        **kwargs: Passed to the provider constructor.

    Returns:
        An ``LLMProvider`` instance.
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
    raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available}")


def available_providers() -> list[str]:
    """Return names of registered LLM providers."""
    return [k for k, _, _ in _PROVIDER_REGISTRY]


def clear_cache() -> None:
    """Clear singleton cache (for testing)."""
    _provider_cache.clear()
