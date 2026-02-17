"""Qdrant vector store — production-grade with native metadata filtering.

Requires the ``qdrant`` extra. Supports both Qdrant Cloud and local instances.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.vectorstore.base import VectorStore
from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        collection_name: str = "investment_research",
        dimension: int = 768,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client required: pip install investment-research-rag[qdrant]"
            ) from exc

        self._qdrant = __import__("qdrant_client")
        self._models = self._qdrant.models
        self._collection_name = collection_name
        self._dimension = dimension

        # Connect to Qdrant
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            # In-memory for testing
            self._client = QdrantClient(":memory:")

        # Ensure collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, dimension)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0

        points = []
        for record in records:
            payload = self._metadata_to_payload(record.metadata)
            payload["text"] = record.text

            points.append(self._models.PointStruct(
                id=record.id,
                vector=record.embedding,
                payload=payload,
            ))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

        logger.info("QdrantStore added %d records", len(records))
        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if metadata_filter:
            conditions = []
            filter_dict = metadata_filter.to_dict()
            for key, value in filter_dict.items():
                conditions.append(
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchValue(value=value),
                    )
                )
            if conditions:
                query_filter = self._models.Filter(must=conditions)

        # qdrant-client >= 1.12 uses query_points; older versions use search
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            metadata = self._payload_to_metadata(payload)
            results.append(SearchResult(
                id=str(point.id),
                text=payload.get("text", ""),
                score=point.score if point.score is not None else 0.0,
                metadata=metadata,
            ))

        return results

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def delete(self, ids: list[str]) -> int:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=self._models.PointIdsList(points=ids),
        )
        return len(ids)

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        from qdrant_client.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
            ),
        )

    # ------------------------------------------------------------------
    # Metadata serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_to_payload(meta: ChunkMetadata) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _payload_to_metadata(payload: dict[str, Any]) -> ChunkMetadata:
        return ChunkMetadata(
            doc_type=DocType(payload.get("doc_type", "other")),
            ticker=payload.get("ticker"),
            filing_date=payload.get("filing_date"),
            section_name=payload.get("section_name"),
            item_number=payload.get("item_number"),
            speaker=payload.get("speaker"),
            page_numbers=payload.get("page_numbers", []),
            source_filename=payload.get("source_filename"),
        )
