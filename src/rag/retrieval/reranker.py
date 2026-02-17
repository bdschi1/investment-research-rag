"""Cross-encoder reranker for search results.

Uses sentence-transformers cross-encoders to re-score and re-sort
retrieval results based on query-document relevance.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.vectorstore.schemas import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required: "
                "pip install investment-research-rag[reranker]"
            ) from exc

        self._model: Any = CrossEncoder(model, device=device)
        self._model_name = model
        logger.info("Loaded reranker model: %s", model)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-score results using cross-encoder and return top_k.

        Args:
            query: The search query.
            results: Initial retrieval results.
            top_k: Number of results to return after reranking.

        Returns:
            Re-sorted list of ``SearchResult`` with updated scores.
        """
        if not results:
            return []

        # Create query-document pairs
        pairs = [(query, r.text) for r in results]

        # Score with cross-encoder
        scores = self._model.predict(pairs)

        # Create new results with cross-encoder scores
        reranked = []
        for result, score in zip(results, scores, strict=True):
            reranked.append(SearchResult(
                id=result.id,
                text=result.text,
                score=float(score),
                metadata=result.metadata,
            ))

        # Sort by cross-encoder score (descending)
        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Reranked %d → %d results (model=%s)",
            len(results),
            min(top_k, len(reranked)),
            self._model_name,
        )

        return reranked[:top_k]
