"""OpenSearch Serverless vector store — AWS-native, serverless.

Requires the ``opensearch`` extra. Uses the OpenSearch Serverless
vector engine via the opensearch-py client with AWS SigV4 authentication.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.chunking.schemas import ChunkMetadata
from rag.documents.schemas import DocType
from rag.vectorstore.base import VectorStore
from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord

logger = logging.getLogger(__name__)


class OpenSearchStore(VectorStore):
    """OpenSearch Serverless vector store with k-NN search."""

    def __init__(
        self,
        collection_endpoint: str | None = None,
        index_name: str = "investment_docs",
        dimension: int = 768,
        region: str = "us-east-1",
    ):
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            import boto3
            from requests_aws4auth import AWS4Auth
        except ImportError as exc:
            raise ImportError(
                "opensearch-py, boto3, and requests-aws4auth required: "
                "pip install investment-research-rag[opensearch]"
            ) from exc

        self._index_name = index_name
        self._dimension = dimension

        if collection_endpoint:
            credentials = boto3.Session().get_credentials()
            auth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                region,
                "aoss",
                session_token=credentials.token,
            )
            self._client = OpenSearch(
                hosts=[{"host": collection_endpoint.replace("https://", ""), "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=60,
            )
        else:
            # Local OpenSearch for development
            self._client = OpenSearch(
                hosts=[{"host": "localhost", "port": 9200}],
                use_ssl=False,
            )

        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the k-NN index if it doesn't exist."""
        if not self._client.indices.exists(self._index_name):
            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self._dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                            },
                        },
                        "text": {"type": "text"},
                        "doc_type": {"type": "keyword"},
                        "ticker": {"type": "keyword"},
                        "filing_date": {"type": "keyword"},
                        "section_name": {"type": "keyword"},
                        "item_number": {"type": "keyword"},
                        "speaker": {"type": "keyword"},
                        "page_numbers": {"type": "integer"},
                        "source_filename": {"type": "keyword"},
                    }
                },
            }
            self._client.indices.create(self._index_name, body=body)
            logger.info(
                "Created OpenSearch index '%s' (dim=%d)", self._index_name, self._dimension
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0

        actions = []
        for record in records:
            doc = self._metadata_to_payload(record.metadata)
            doc["text"] = record.text
            doc["embedding"] = record.embedding

            actions.append({"index": {"_index": self._index_name, "_id": record.id}})
            actions.append(doc)

        if actions:
            body = "\n".join([json.dumps(a) for a in actions]) + "\n"
            self._client.bulk(body=body)
            self._client.indices.refresh(self._index_name)

        logger.info("OpenSearchStore added %d records", len(records))
        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        query_body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
        }

        # Add metadata filter if provided
        if metadata_filter:
            filter_clauses = []
            filter_dict = metadata_filter.to_dict()
            for field_name, value in filter_dict.items():
                filter_clauses.append({"term": {field_name: value}})

            if filter_clauses:
                query_body["query"] = {
                    "bool": {
                        "must": [query_body["query"]],
                        "filter": filter_clauses,
                    }
                }

        response = self._client.search(index=self._index_name, body=query_body)

        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            metadata = self._payload_to_metadata(source)
            results.append(
                SearchResult(
                    id=hit["_id"],
                    text=source.get("text", ""),
                    score=hit["_score"] if hit["_score"] is not None else 0.0,
                    metadata=metadata,
                )
            )

        return results

    def count(self) -> int:
        try:
            response = self._client.count(index=self._index_name)
            return response["count"]
        except Exception:
            return 0

    def delete(self, ids: list[str]) -> int:
        actions = []
        for doc_id in ids:
            actions.append({"delete": {"_index": self._index_name, "_id": doc_id}})

        if actions:
            body = "\n".join([json.dumps(a) for a in actions]) + "\n"
            self._client.bulk(body=body)
            self._client.indices.refresh(self._index_name)

        return len(ids)

    def clear(self) -> None:
        if self._client.indices.exists(self._index_name):
            self._client.indices.delete(self._index_name)
        self._ensure_index()

    # ------------------------------------------------------------------
    # Metadata serialization (matches QdrantStore pattern)
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
