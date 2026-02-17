"""Lambda handler for document ingestion — triggered by S3 -> SQS.

Thin wrapper around IngestPipeline. All business logic lives in src/rag/.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import boto3

from rag.config import load_settings
from rag.documents.schemas import DocType
from rag.embeddings.factory import get_embedding_provider
from rag.pipeline.ingest import IngestPipeline
from rag.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

s3_client = boto3.client("s3")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process S3 events from SQS — download file, ingest, return result."""
    settings = load_settings()

    emb = get_embedding_provider(
        settings.embedding.provider,
        model=settings.embedding.model,
    )
    store = get_vector_store(
        settings.vectorstore.backend,
        collection_endpoint=settings.vectorstore.opensearch_endpoint,
        index_name=settings.vectorstore.collection,
        dimension=settings.embedding.dimension,
        region=settings.vectorstore.opensearch_region,
    )
    pipeline = IngestPipeline(embedding_provider=emb, vector_store=store)

    results = []
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        for s3_record in body.get("Records", []):
            bucket = s3_record["s3"]["bucket"]["name"]
            key = unquote_plus(s3_record["s3"]["object"]["key"])

            # S3 key convention: ticker/doc_type/filename.pdf
            parts = key.split("/")
            ticker = parts[0] if len(parts) > 2 else None
            doc_type_str = parts[1] if len(parts) > 2 else "other"

            with tempfile.NamedTemporaryFile(suffix=Path(key).suffix, delete=False) as tmp:
                s3_client.download_file(bucket, key, tmp.name)
                result = pipeline.ingest_file(
                    Path(tmp.name),
                    doc_type=DocType(doc_type_str),
                    ticker=ticker,
                )

            results.append({
                "source": key,
                "chunks_created": result.chunks_created,
                "chunks_stored": result.chunks_stored,
                "warnings": result.warnings,
            })

    return {"statusCode": 200, "results": results}
