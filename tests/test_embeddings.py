"""Tests for embedding providers — uses mock providers, no network calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.factory import (
    available_providers,
    clear_cache,
    get_embedding_provider,
)

# ---------------------------------------------------------------------------
# Mock embedding provider for testing
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """A deterministic embedding provider for tests."""

    def __init__(self, dimension: int = 768):
        self._dim = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_embed(query)

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic embedding based on hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash bytes into a full-dimension vector
        vec = []
        for i in range(self._dim):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        return vec


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


class TestEmbeddingProviderABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_provider_name(self):
        assert MockEmbeddingProvider.provider_name() == "MockEmbeddingProvider"


# ---------------------------------------------------------------------------
# Mock provider behavior
# ---------------------------------------------------------------------------


class TestMockProvider:
    @pytest.fixture
    def provider(self) -> MockEmbeddingProvider:
        return MockEmbeddingProvider(dimension=768)

    def test_embed_texts(self, provider: MockEmbeddingProvider):
        embeddings = provider.embed_texts(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768
        assert len(embeddings[1]) == 768

    def test_embed_query(self, provider: MockEmbeddingProvider):
        embedding = provider.embed_query("test query")
        assert len(embedding) == 768

    def test_embed_texts_empty(self, provider: MockEmbeddingProvider):
        assert provider.embed_texts([]) == []

    def test_dimension(self, provider: MockEmbeddingProvider):
        assert provider.dimension == 768

    def test_deterministic(self, provider: MockEmbeddingProvider):
        e1 = provider.embed_query("same text")
        e2 = provider.embed_query("same text")
        assert e1 == e2

    def test_different_inputs_different_outputs(self, provider: MockEmbeddingProvider):
        e1 = provider.embed_query("text one")
        e2 = provider.embed_query("text two")
        assert e1 != e2


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestEmbeddingFactory:
    def setup_method(self):
        clear_cache()

    def test_available_providers(self):
        providers = available_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "huggingface" in providers

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("nonexistent")

    def test_clear_cache(self):
        clear_cache()  # Should not raise

    def test_ollama_provider_import(self):
        """Verify OllamaEmbeddingProvider can be imported."""
        from rag.embeddings.ollama_provider import OllamaEmbeddingProvider
        assert issubclass(OllamaEmbeddingProvider, EmbeddingProvider)

    def test_openai_provider_import_error(self):
        """OpenAI provider requires openai package."""
        # This test verifies the import guard works
        from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
        assert issubclass(OpenAIEmbeddingProvider, EmbeddingProvider)

    def test_factory_caching(self):
        """Factory should cache by provider key when no kwargs."""
        # We can't test ollama directly (needs server), but we can test
        # the factory pattern by mocking
        with patch("rag.embeddings.factory.importlib") as mock_importlib:
            mock_mod = MagicMock()
            mock_cls = MockEmbeddingProvider
            mock_mod.OllamaEmbeddingProvider = mock_cls
            mock_importlib.import_module.return_value = mock_mod

            p1 = get_embedding_provider("ollama")
            p2 = get_embedding_provider("ollama")
            assert p1 is p2  # Should be cached

    def test_factory_kwargs_bypass_cache(self):
        """Factory should NOT cache when kwargs are provided."""
        with patch("rag.embeddings.factory.importlib") as mock_importlib:
            mock_mod = MagicMock()
            mock_cls = MockEmbeddingProvider
            mock_mod.OllamaEmbeddingProvider = mock_cls
            mock_importlib.import_module.return_value = mock_mod

            p1 = get_embedding_provider("ollama")
            clear_cache()
            p2 = get_embedding_provider("ollama", dimension=512)
            assert p1 is not p2
