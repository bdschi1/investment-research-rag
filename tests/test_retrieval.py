"""Tests for retrieval and LLM layers — uses mock providers, no network."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.embeddings.base import EmbeddingProvider
from rag.llm.base import LLMProvider
from rag.llm.factory import available_providers as llm_providers
from rag.retrieval.retriever import Retriever
from rag.retrieval.schemas import RetrievalConfig, RetrievalResult
from rag.vectorstore.faiss_store import FAISSStore
from rag.vectorstore.schemas import MetadataFilter, VectorRecord

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

DIM = 64


class MockEmbedder(EmbeddingProvider):
    """Deterministic embeddings for testing."""

    def __init__(self, dim: int = DIM):
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_embed(query)

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = np.array([h[i % len(h)] / 255.0 for i in range(self._dim)], dtype=np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()


class MockLLM(LLMProvider):
    """Mock LLM that echoes context for testing."""

    def __init__(self, model: str = "mock-llm"):
        self.model = model
        self.last_prompt = ""

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        # Echo back a mock answer with citation references
        return (
            "Based on the provided context, revenue grew 6% year-over-year [1]. "
            "The services segment reached a new record [2]. "
            "Risk factors include foreign exchange headwinds [3]."
        )


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


class TestRetriever:
    @pytest.fixture
    def setup(self):
        embedder = MockEmbedder(dim=DIM)
        store = FAISSStore(dimension=DIM)

        # Pre-populate with records
        texts = [
            "Revenue grew 6% to $94.9 billion",
            "Services reached record $25 billion",
            "Foreign exchange headwinds impacted revenue",
            "iPhone 16 demand exceeded expectations",
            "Gross margin expanded to 46.2%",
        ]
        records = []
        for i, text in enumerate(texts):
            emb = embedder.embed_texts([text])[0]
            records.append(VectorRecord(
                id=f"doc-{i}",
                text=text,
                embedding=emb,
                metadata=ChunkMetadata(
                    ticker="AAPL",
                    doc_type=DocType.EARNINGS_TRANSCRIPT,
                    section_name=f"section_{i}",
                ),
            ))
        store.add(records)

        retriever = Retriever(
            embedding_provider=embedder,
            vector_store=store,
        )

        return retriever, embedder, store

    def test_basic_retrieval(self, setup):
        retriever, embedder, store = setup
        result = retriever.retrieve("What was the revenue?")
        assert isinstance(result, RetrievalResult)
        assert len(result.results) > 0
        assert result.query == "What was the revenue?"

    def test_retrieval_with_config(self, setup):
        retriever, _, _ = setup
        config = RetrievalConfig(top_k=2)
        result = retriever.retrieve("revenue growth", config=config)
        assert len(result.results) <= 2

    def test_retrieval_with_metadata_filter(self, setup):
        retriever, _, _ = setup
        config = RetrievalConfig(
            top_k=10,
            metadata_filter=MetadataFilter(ticker="AAPL"),
        )
        result = retriever.retrieve("revenue", config=config)
        assert all(r.metadata.ticker == "AAPL" for r in result.results)

    def test_retrieval_empty_store(self):
        embedder = MockEmbedder()
        store = FAISSStore(dimension=DIM)
        retriever = Retriever(embedder, store)
        result = retriever.retrieve("anything")
        assert result.results == []

    def test_retrieval_result_count(self, setup):
        retriever, _, _ = setup
        result = retriever.retrieve("Apple")
        assert result.total_candidates > 0


# ---------------------------------------------------------------------------
# LLM factory tests
# ---------------------------------------------------------------------------


class TestLLMFactory:
    def test_available_providers(self):
        providers = llm_providers()
        assert "ollama" in providers
        assert "anthropic" in providers
        assert "openai" in providers

    def test_mock_llm(self):
        llm = MockLLM()
        response = llm.generate("test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_llm_with_system(self):
        llm = MockLLM()
        response = llm.generate("test", system="You are an analyst.")
        assert isinstance(response, str)

    def test_llm_base_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_provider_name(self):
        assert MockLLM.provider_name() == "MockLLM"


# ---------------------------------------------------------------------------
# RetrievalConfig / RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalSchemas:
    def test_default_config(self):
        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.rerank is False
        assert config.min_score == 0.0

    def test_custom_config(self):
        config = RetrievalConfig(top_k=5, rerank=True, rerank_top_k=3, min_score=0.5)
        assert config.top_k == 5
        assert config.rerank is True
        assert config.rerank_top_k == 3

    def test_retrieval_result(self):
        result = RetrievalResult(query="test", results=[], total_candidates=0)
        assert result.query == "test"
        assert result.reranked is False
