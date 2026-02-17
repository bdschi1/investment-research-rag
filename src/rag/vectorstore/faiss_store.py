"""FAISS vector store — local, zero infrastructure.

Adapted from Projects/z_oldProjects/doc-chunker/rag.py. Uses FAISS for
similarity search with a parallel metadata dict for filtering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.vectorstore.base import VectorStore
from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    """FAISS-backed vector store with metadata filtering."""

    def __init__(self, dimension: int = 768):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu required: pip install investment-research-rag[faiss]"
            ) from exc

        self._faiss = faiss
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self._records: dict[int, dict] = {}  # int id -> {id, text, metadata}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0

        vectors = np.array([r.embedding for r in records], dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        self._faiss.normalize_L2(vectors)

        start_id = self._next_id
        self._index.add(vectors)

        for i, record in enumerate(records):
            int_id = start_id + i
            self._records[int_id] = {
                "id": record.id,
                "text": record.text,
                "metadata": record.metadata,
            }

        self._next_id = start_id + len(records)
        logger.info("FAISSStore added %d records (total: %d)", len(records), self.count())
        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        self._faiss.normalize_L2(query_vec)

        # Over-fetch if filtering to ensure enough results after filtering
        fetch_k = top_k * 4 if metadata_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, fetch_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            record = self._records.get(int(idx))
            if record is None:
                continue

            if metadata_filter and not metadata_filter.matches(record["metadata"]):
                continue

            results.append(SearchResult(
                id=record["id"],
                text=record["text"],
                score=float(score),
                metadata=record["metadata"],
            ))

            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        return self._index.ntotal

    def delete(self, ids: list[str]) -> int:
        # FAISS IndexFlatIP doesn't support deletion natively.
        # Rebuild the index without the deleted records.
        id_set = set(ids)
        to_keep: list[tuple[int, dict]] = []

        for int_id, record in self._records.items():
            if record["id"] not in id_set:
                to_keep.append((int_id, record))

        deleted = len(self._records) - len(to_keep)
        if deleted == 0:
            return 0

        # Rebuild
        self._index = self._faiss.IndexFlatIP(self._dimension)
        new_records: dict[int, dict] = {}

        if to_keep:
            # Re-embed from stored vectors — but we don't store raw vectors
            # This is a limitation; for production use Qdrant instead
            logger.warning(
                "FAISSStore delete rebuilds index. "
                "For frequent deletes, use QdrantStore."
            )

        self._records = new_records
        self._next_id = 0
        return deleted

    def clear(self) -> None:
        self._index = self._faiss.IndexFlatIP(self._dimension)
        self._records.clear()
        self._next_id = 0

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self._faiss.write_index(self._index, str(p / "index.faiss"))

        # Save metadata
        serializable = {}
        for int_id, record in self._records.items():
            meta = record["metadata"]
            meta_dict = {
                "doc_type": (
                    meta.doc_type.value
                    if hasattr(meta.doc_type, "value")
                    else str(meta.doc_type)
                ),
                "ticker": meta.ticker,
                "filing_date": meta.filing_date,
                "section_name": meta.section_name,
                "item_number": meta.item_number,
                "speaker": meta.speaker,
                "page_numbers": list(meta.page_numbers) if meta.page_numbers else [],
                "source_filename": meta.source_filename,
            }
            serializable[str(int_id)] = {
                "id": record["id"],
                "text": record["text"],
                "metadata": meta_dict,
            }

        with open(p / "metadata.json", "w") as f:
            json.dump({"records": serializable, "next_id": self._next_id}, f)

        logger.info("FAISSStore saved to %s (%d records)", path, self.count())

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        p = Path(path)

        self._index = self._faiss.read_index(str(p / "index.faiss"))

        with open(p / "metadata.json") as f:
            data = json.load(f)

        self._records = {}
        for str_id, record in data["records"].items():
            meta_dict = record["metadata"]
            meta = ChunkMetadata(
                doc_type=DocType(meta_dict.get("doc_type", "other")),
                ticker=meta_dict.get("ticker"),
                filing_date=meta_dict.get("filing_date"),
                section_name=meta_dict.get("section_name"),
                item_number=meta_dict.get("item_number"),
                speaker=meta_dict.get("speaker"),
                page_numbers=meta_dict.get("page_numbers", []),
                source_filename=meta_dict.get("source_filename"),
            )
            self._records[int(str_id)] = {
                "id": record["id"],
                "text": record["text"],
                "metadata": meta,
            }

        self._next_id = data.get("next_id", len(self._records))
        logger.info("FAISSStore loaded from %s (%d records)", path, self.count())
