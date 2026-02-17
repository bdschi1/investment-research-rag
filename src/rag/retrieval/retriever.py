"""Retriever — embed query, search vector store, optionally rerank."""

from __future__ import annotations

import logging

from rag.embeddings.base import EmbeddingProvider
from rag.retrieval.schemas import RetrievalConfig, RetrievalResult
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates embedding → search → (optional) reranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: object | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """Run a full retrieval: embed → search → filter → rerank.

        Args:
            query: The search query.
            config: Retrieval settings (top_k, rerank, filters).

        Returns:
            A ``RetrievalResult`` with ranked search results.
        """
        cfg = config or RetrievalConfig()

        # Step 1: Embed the query
        query_embedding = self.embedding_provider.embed_query(query)

        # Step 2: Search the vector store
        fetch_k = cfg.top_k * 3 if cfg.rerank else cfg.top_k
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            metadata_filter=cfg.metadata_filter,
        )

        total_candidates = len(raw_results)

        # Step 3: Apply minimum score filter
        if cfg.min_score > 0:
            raw_results = [r for r in raw_results if r.score >= cfg.min_score]

        # Step 4: Rerank (if enabled and reranker available)
        reranked = False
        if cfg.rerank and self.reranker is not None:
            raw_results = self._rerank(query, raw_results, cfg.rerank_top_k)
            reranked = True
        else:
            raw_results = raw_results[: cfg.top_k]

        logger.info(
            "Retrieved %d results for query (candidates=%d, reranked=%s)",
            len(raw_results),
            total_candidates,
            reranked,
        )

        return RetrievalResult(
            query=query,
            results=raw_results,
            total_candidates=total_candidates,
            reranked=reranked,
        )

    def _rerank(self, query: str, results, top_k: int):
        """Rerank results using a cross-encoder."""
        from rag.retrieval.reranker import Reranker

        if not isinstance(self.reranker, Reranker):
            return results[:top_k]

        return self.reranker.rerank(query, results, top_k=top_k)
