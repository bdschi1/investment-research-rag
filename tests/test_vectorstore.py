"""Tests for vector store backends — FAISS and Qdrant (in-memory)."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.vectorstore.base import VectorStore
from rag.vectorstore.factory import available_stores, clear_cache, get_vector_store
from rag.vectorstore.faiss_store import FAISSStore
from rag.vectorstore.qdrant_store import QdrantStore
from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64  # Small dimension for fast tests


def _random_embedding(dim: int = DIM) -> list[float]:
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)  # Normalize
    return vec.tolist()


def _make_record(
    text: str = "sample text",
    ticker: str | None = None,
    doc_type: DocType = DocType.OTHER,
    section_name: str | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=str(uuid.uuid4()),
        text=text,
        embedding=_random_embedding(),
        metadata=ChunkMetadata(
            doc_type=doc_type,
            ticker=ticker,
            section_name=section_name,
        ),
    )


# ---------------------------------------------------------------------------
# FAISS Store Tests
# ---------------------------------------------------------------------------


class TestFAISSStore:
    @pytest.fixture
    def store(self) -> FAISSStore:
        return FAISSStore(dimension=DIM)

    def test_is_vector_store(self):
        assert issubclass(FAISSStore, VectorStore)

    def test_add_records(self, store: FAISSStore):
        records = [_make_record(f"doc {i}") for i in range(5)]
        count = store.add(records)
        assert count == 5
        assert store.count() == 5

    def test_add_empty(self, store: FAISSStore):
        assert store.add([]) == 0
        assert store.count() == 0

    def test_search_basic(self, store: FAISSStore):
        # Add a record and search for it
        record = _make_record("revenue grew 10%")
        store.add([record])

        results = store.search(record.embedding, top_k=5)
        assert len(results) == 1
        assert results[0].text == "revenue grew 10%"
        assert results[0].score > 0.9  # Should be near-perfect match

    def test_search_top_k(self, store: FAISSStore):
        records = [_make_record(f"doc {i}") for i in range(10)]
        store.add(records)

        results = store.search(_random_embedding(), top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, store: FAISSStore):
        results = store.search(_random_embedding(), top_k=5)
        assert results == []

    def test_search_with_metadata_filter(self, store: FAISSStore):
        records = [
            _make_record("AAPL revenue", ticker="AAPL"),
            _make_record("MSFT revenue", ticker="MSFT"),
            _make_record("GOOG revenue", ticker="GOOG"),
        ]
        store.add(records)

        # Search with ticker filter
        mf = MetadataFilter(ticker="AAPL")
        results = store.search(records[0].embedding, top_k=10, metadata_filter=mf)
        assert all(r.metadata.ticker == "AAPL" for r in results)
        assert len(results) == 1

    def test_search_with_doc_type_filter(self, store: FAISSStore):
        records = [
            _make_record("10-K filing", doc_type=DocType.SEC_FILING),
            _make_record("research report", doc_type=DocType.RESEARCH_REPORT),
        ]
        store.add(records)

        mf = MetadataFilter(doc_type="sec_filing")
        results = store.search(records[0].embedding, top_k=10, metadata_filter=mf)
        assert len(results) == 1
        assert results[0].metadata.doc_type == DocType.SEC_FILING

    def test_clear(self, store: FAISSStore):
        store.add([_make_record() for _ in range(5)])
        assert store.count() == 5
        store.clear()
        assert store.count() == 0

    def test_save_and_load(self, store: FAISSStore, tmp_path: Path):
        records = [
            _make_record("AAPL analysis", ticker="AAPL", doc_type=DocType.SEC_FILING),
            _make_record("MSFT analysis", ticker="MSFT", doc_type=DocType.RESEARCH_REPORT),
        ]
        store.add(records)

        # Save
        save_path = str(tmp_path / "faiss_test")
        store.save(save_path)

        # Load into new store
        new_store = FAISSStore(dimension=DIM)
        new_store.load(save_path)

        assert new_store.count() == 2
        results = new_store.search(records[0].embedding, top_k=5)
        assert len(results) == 2
        # Verify metadata survived serialization
        tickers = {r.metadata.ticker for r in results}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_store_name(self):
        assert FAISSStore.store_name() == "FAISSStore"

    def test_search_returns_sorted_by_score(self, store: FAISSStore):
        records = [_make_record(f"doc {i}") for i in range(10)]
        store.add(records)
        results = store.search(_random_embedding(), top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Qdrant Store Tests
# ---------------------------------------------------------------------------


class TestQdrantStore:
    @pytest.fixture
    def store(self) -> QdrantStore:
        return QdrantStore(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            dimension=DIM,
        )

    def test_is_vector_store(self):
        assert issubclass(QdrantStore, VectorStore)

    def test_add_records(self, store: QdrantStore):
        records = [_make_record(f"doc {i}") for i in range(5)]
        count = store.add(records)
        assert count == 5
        assert store.count() == 5

    def test_add_empty(self, store: QdrantStore):
        assert store.add([]) == 0

    def test_search_basic(self, store: QdrantStore):
        record = _make_record("revenue grew 10%")
        store.add([record])

        results = store.search(record.embedding, top_k=5)
        assert len(results) == 1
        assert results[0].text == "revenue grew 10%"
        assert results[0].score > 0.9

    def test_search_top_k(self, store: QdrantStore):
        records = [_make_record(f"doc {i}") for i in range(10)]
        store.add(records)

        results = store.search(_random_embedding(), top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, store: QdrantStore):
        results = store.search(_random_embedding(), top_k=5)
        assert results == []

    def test_search_with_metadata_filter(self, store: QdrantStore):
        records = [
            _make_record("AAPL revenue", ticker="AAPL"),
            _make_record("MSFT revenue", ticker="MSFT"),
            _make_record("GOOG revenue", ticker="GOOG"),
        ]
        store.add(records)

        mf = MetadataFilter(ticker="AAPL")
        results = store.search(records[0].embedding, top_k=10, metadata_filter=mf)
        assert all(r.metadata.ticker == "AAPL" for r in results)

    def test_search_with_doc_type_filter(self, store: QdrantStore):
        records = [
            _make_record("filing", doc_type=DocType.SEC_FILING),
            _make_record("report", doc_type=DocType.RESEARCH_REPORT),
        ]
        store.add(records)

        mf = MetadataFilter(doc_type="sec_filing")
        results = store.search(records[0].embedding, top_k=10, metadata_filter=mf)
        assert len(results) == 1

    def test_delete(self, store: QdrantStore):
        records = [_make_record(f"doc {i}") for i in range(3)]
        store.add(records)
        assert store.count() == 3

        store.delete([records[0].id])
        assert store.count() == 2

    def test_clear(self, store: QdrantStore):
        store.add([_make_record() for _ in range(5)])
        assert store.count() == 5
        store.clear()
        assert store.count() == 0

    def test_store_name(self):
        assert QdrantStore.store_name() == "QdrantStore"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestVectorSchemas:
    def test_vector_record(self):
        rec = VectorRecord(
            id="test-1",
            text="hello",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(ticker="AAPL"),
        )
        assert rec.id == "test-1"
        assert rec.metadata.ticker == "AAPL"

    def test_search_result_frozen(self):
        sr = SearchResult(
            id="test-1",
            text="hello",
            score=0.95,
            metadata=ChunkMetadata(ticker="AAPL"),
        )
        with pytest.raises(AttributeError):
            sr.score = 0.5  # type: ignore[misc]

    def test_metadata_filter_matches(self):
        meta = ChunkMetadata(
            doc_type=DocType.SEC_FILING,
            ticker="AAPL",
            section_name="Risk Factors",
        )
        assert MetadataFilter(ticker="AAPL").matches(meta)
        assert MetadataFilter(doc_type="sec_filing").matches(meta)
        assert not MetadataFilter(ticker="MSFT").matches(meta)
        assert MetadataFilter().matches(meta)  # No filter = match all

    def test_metadata_filter_to_dict(self):
        mf = MetadataFilter(ticker="AAPL", doc_type="sec_filing")
        d = mf.to_dict()
        assert d == {"ticker": "AAPL", "doc_type": "sec_filing"}

    def test_metadata_filter_empty_to_dict(self):
        assert MetadataFilter().to_dict() == {}


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestVectorStoreFactory:
    def setup_method(self):
        clear_cache()

    def test_available_stores(self):
        stores = available_stores()
        assert "faiss" in stores
        assert "qdrant" in stores

    def test_get_faiss(self):
        store = get_vector_store("faiss", dimension=DIM)
        assert isinstance(store, FAISSStore)

    def test_get_qdrant(self):
        store = get_vector_store("qdrant", dimension=DIM)
        assert isinstance(store, QdrantStore)

    def test_unknown_store_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store"):
            get_vector_store("pinecone")

    def test_caching(self):
        s1 = get_vector_store("faiss")
        s2 = get_vector_store("faiss")
        assert s1 is s2

    def test_kwargs_bypass_cache(self):
        s1 = get_vector_store("faiss")
        clear_cache()
        s2 = get_vector_store("faiss", dimension=512)
        assert s1 is not s2
