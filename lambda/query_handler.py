"""Lambda handler for RAG queries — triggered by API Gateway.

Thin wrapper around QueryPipeline. All business logic lives in src/rag/.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from rag.config import load_settings
from rag.embeddings.factory import get_embedding_provider
from rag.llm.factory import get_llm_provider
from rag.pipeline.query import QueryPipeline
from rag.pipeline.schemas import RAGQuery
from rag.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Initialize outside handler for Lambda warm-start reuse
_pipeline: QueryPipeline | None = None


def _get_pipeline() -> QueryPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

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
    llm = get_llm_provider(
        settings.llm.provider,
        model=settings.llm.model,
    )
    _pipeline = QueryPipeline(
        embedding_provider=emb, vector_store=store, llm_provider=llm
    )
    return _pipeline


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle API Gateway request — parse query, run pipeline, return JSON."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        body = {}

    question = body.get("question", "")

    if not question:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Missing 'question' field"}),
        }

    rag_query = RAGQuery(
        question=question,
        ticker=body.get("ticker"),
        doc_type=body.get("doc_type"),
        top_k=body.get("top_k", 10),
    )

    pipeline = _get_pipeline()
    response = pipeline.query(rag_query)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "question": response.question,
            "answer": response.answer,
            "citations": [
                {
                    "index": c.index,
                    "source": c.source,
                    "section": c.section,
                    "ticker": c.ticker,
                }
                for c in response.citations
            ],
            "model": response.model,
            "retrieval_count": response.retrieval_count,
        }),
    }
