"""Retrieval quality metrics — precision@k, recall@k, MRR, nDCG."""

from __future__ import annotations

import math


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Precision@k — fraction of top-k results that are relevant.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ordered list of retrieved document IDs.
        k: Cutoff position.

    Returns:
        Precision score (0.0 to 1.0).
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Recall@k — fraction of relevant documents found in top-k.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ordered list of retrieved document IDs.
        k: Cutoff position.

    Returns:
        Recall score (0.0 to 1.0).
    """
    if not relevant or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    """Mean Reciprocal Rank — inverse of rank of first relevant result.

    Args:
        relevant: Set of relevant document IDs.
        retrieved: Ordered list of retrieved document IDs.

    Returns:
        MRR score (0.0 to 1.0).
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    relevance_scores: list[float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at position k.

    Args:
        relevance_scores: Relevance scores in retrieval order.
        k: Cutoff position.

    Returns:
        nDCG score (0.0 to 1.0).
    """
    if not relevance_scores or k <= 0:
        return 0.0

    scores = relevance_scores[:k]

    # DCG
    dcg = scores[0]
    for i in range(1, len(scores)):
        dcg += scores[i] / math.log2(i + 1)

    # Ideal DCG (sort scores descending)
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = ideal[0]
    for i in range(1, len(ideal)):
        idcg += ideal[i] / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg
