"""Tests for the end-to-end RAG pipeline — fully mocked, no network."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.embeddings.base import EmbeddingProvider
from rag.llm.base import LLMProvider
from rag.pipeline.citations import extract_citations, format_citations
from rag.pipeline.ingest import IngestPipeline
from rag.pipeline.prompts import build_rag_prompt, format_context
from rag.pipeline.query import QueryPipeline
from rag.pipeline.schemas import Citation, IngestResult, RAGQuery, RAGResponse
from rag.vectorstore.faiss_store import FAISSStore
from rag.vectorstore.schemas import SearchResult

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

DIM = 64


class MockEmbedder(EmbeddingProvider):
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
    def __init__(self):
        self.model = "mock-llm"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return (
            "Revenue grew 6% year-over-year [1]. The services segment "
            "reached a new record of $25 billion [2]."
        )


# ---------------------------------------------------------------------------
# Citation tests
# ---------------------------------------------------------------------------


class TestCitations:
    def test_extract_simple(self):
        answer = "Revenue grew [1]. Services were strong [2]."
        results = [
            SearchResult(
                id="a", text="Revenue text", score=0.9,
                metadata=ChunkMetadata(source_filename="10k.pdf"),
            ),
            SearchResult(
                id="b", text="Services text", score=0.8,
                metadata=ChunkMetadata(source_filename="10k.pdf"),
            ),
        ]
        citations = extract_citations(answer, results)
        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[1].index == 2

    def test_extract_range(self):
        answer = "Multiple factors affected results [1-3]."
        results = [
            SearchResult(id="a", text="A", score=0.9, metadata=ChunkMetadata()),
            SearchResult(id="b", text="B", score=0.8, metadata=ChunkMetadata()),
            SearchResult(id="c", text="C", score=0.7, metadata=ChunkMetadata()),
        ]
        citations = extract_citations(answer, results)
        assert len(citations) == 3

    def test_extract_comma_list(self):
        answer = "Key metrics include [1,3]."
        results = [
            SearchResult(id="a", text="A", score=0.9, metadata=ChunkMetadata()),
            SearchResult(id="b", text="B", score=0.8, metadata=ChunkMetadata()),
            SearchResult(id="c", text="C", score=0.7, metadata=ChunkMetadata()),
        ]
        citations = extract_citations(answer, results)
        assert len(citations) == 2
        indices = {c.index for c in citations}
        assert indices == {1, 3}

    def test_extract_out_of_range(self):
        answer = "This is [99] which doesn't exist."
        results = [
            SearchResult(id="a", text="A", score=0.9, metadata=ChunkMetadata()),
        ]
        citations = extract_citations(answer, results)
        assert len(citations) == 0

    def test_extract_no_citations(self):
        answer = "No citations in this answer."
        citations = extract_citations(answer, [])
        assert citations == []

    def test_format_citations(self):
        citations = [
            Citation(
                index=1, text="Revenue", source="10k.pdf",
                section="MD&A", ticker="AAPL", score=0.9,
            ),
            Citation(
                index=2, text="Services", source="10k.pdf",
                section="Business", ticker="AAPL", score=0.8,
            ),
        ]
        formatted = format_citations(citations)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "AAPL" in formatted
        assert "Sources:" in formatted

    def test_format_empty_citations(self):
        assert format_citations([]) == ""


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_format_context(self):
        texts = ["Revenue data.", "Services data."]
        result = format_context(texts)
        assert "[1]" in result
        assert "[2]" in result
        assert "Revenue data." in result

    def test_format_context_with_sources(self):
        texts = ["Data"]
        sources = ["10k.pdf"]
        result = format_context(texts, sources)
        assert "[1]" in result
        assert "10k.pdf" in result

    def test_build_rag_prompt(self):
        prompt = build_rag_prompt(
            question="What was revenue?",
            context_texts=["Revenue was $94.9B"],
        )
        assert "What was revenue?" in prompt
        assert "Revenue was $94.9B" in prompt
        assert "[1]" in prompt


# ---------------------------------------------------------------------------
# Ingest pipeline tests
# ---------------------------------------------------------------------------


class TestIngestPipeline:
    @pytest.fixture
    def pipeline(self):
        return IngestPipeline(
            embedding_provider=MockEmbedder(dim=DIM),
            vector_store=FAISSStore(dimension=DIM),
        )

    def test_ingest_text(self, pipeline: IngestPipeline):
        result = pipeline.ingest_text(
            "Apple reported revenue of $94.9 billion. "
            "Services grew 14% to $25 billion. " * 10,
            source_name="test",
            ticker="AAPL",
        )
        assert isinstance(result, IngestResult)
        assert result.chunks_created > 0
        assert result.chunks_embedded > 0
        assert result.chunks_stored > 0

    def test_ingest_empty_text(self, pipeline: IngestPipeline):
        result = pipeline.ingest_text("", source_name="empty")
        assert result.chunks_created == 0

    def test_ingest_file(self, pipeline: IngestPipeline, sample_txt_file):
        result = pipeline.ingest_file(
            sample_txt_file,
            doc_type=DocType.RESEARCH_REPORT,
            ticker="AAPL",
        )
        assert result.chunks_created > 0
        assert result.chunks_stored > 0

    def test_ingest_stores_in_vector_store(self, pipeline: IngestPipeline):
        pipeline.ingest_text(
            "Revenue data. " * 50,
            source_name="test",
        )
        assert pipeline.vector_store.count() > 0


# ---------------------------------------------------------------------------
# Query pipeline tests
# ---------------------------------------------------------------------------


class TestQueryPipeline:
    @pytest.fixture
    def pipeline(self):
        embedder = MockEmbedder(dim=DIM)
        store = FAISSStore(dimension=DIM)

        # Pre-populate store
        ingestor = IngestPipeline(
            embedding_provider=embedder,
            vector_store=store,
        )
        ingestor.ingest_text(
            "Apple Inc. reported revenue of $94.9 billion, up 6% year-over-year. "
            "The services segment reached a new record of $25.0 billion. "
            "Gross margin expanded 120 basis points to 46.2%. "
            "iPhone revenue was $69.9 billion, driven by strong iPhone 16 demand. " * 5,
            source_name="aapl_q4.txt",
            ticker="AAPL",
        )

        return QueryPipeline(
            embedding_provider=embedder,
            vector_store=store,
            llm_provider=MockLLM(),
        )

    def test_basic_query(self, pipeline: QueryPipeline):
        response = pipeline.query_simple("What was Apple's revenue?")
        assert isinstance(response, RAGResponse)
        assert response.answer
        assert response.question == "What was Apple's revenue?"
        assert response.model == "mock-llm"

    def test_query_with_filters(self, pipeline: QueryPipeline):
        query = RAGQuery(
            question="What was revenue?",
            ticker="AAPL",
            top_k=3,
        )
        response = pipeline.query(query)
        assert response.retrieval_count > 0

    def test_query_returns_citations(self, pipeline: QueryPipeline):
        response = pipeline.query_simple("revenue growth")
        # MockLLM includes [1] and [2] in its output
        assert len(response.citations) > 0

    def test_query_returns_context(self, pipeline: QueryPipeline):
        response = pipeline.query_simple("revenue")
        assert len(response.context_texts) > 0

    def test_query_empty_store(self):
        pipeline = QueryPipeline(
            embedding_provider=MockEmbedder(dim=DIM),
            vector_store=FAISSStore(dimension=DIM),
            llm_provider=MockLLM(),
        )
        response = pipeline.query_simple("anything")
        assert "No relevant documents" in response.answer


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPipelineSchemas:
    def test_rag_query(self):
        q = RAGQuery(question="What was revenue?", ticker="AAPL", top_k=5)
        assert q.question == "What was revenue?"
        assert q.rerank is False

    def test_rag_response(self):
        r = RAGResponse(
            question="test",
            answer="Revenue was $94.9B [1].",
            model="gpt-4",
            retrieval_count=3,
        )
        assert r.model == "gpt-4"

    def test_ingest_result(self):
        r = IngestResult(
            source="file.pdf",
            chunks_created=10,
            chunks_embedded=10,
            chunks_stored=10,
        )
        assert r.warnings == []

    def test_citation(self):
        c = Citation(
            index=1,
            text="Revenue data",
            source="10k.pdf",
            section="MD&A",
            ticker="AAPL",
            score=0.95,
        )
        assert c.index == 1
