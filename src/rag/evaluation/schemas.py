"""Data models for RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LikertScore(int, Enum):
    """5-point Likert scale for answer quality."""

    VERY_POOR = 1
    POOR = 2
    ADEQUATE = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class EvalScenario:
    """A single evaluation scenario."""

    id: str
    question: str
    expected_answer: str | None = None
    expected_citations: list[str] = field(default_factory=list)
    ticker: str | None = None
    doc_type: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single scenario."""

    scenario_id: str
    question: str
    generated_answer: str
    retrieval_count: int = 0
    citation_count: int = 0

    # Quality scores
    relevance: LikertScore = LikertScore.ADEQUATE
    accuracy: LikertScore = LikertScore.ADEQUATE
    completeness: LikertScore = LikertScore.ADEQUATE
    citation_quality: LikertScore = LikertScore.ADEQUATE

    # Retrieval metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0


@dataclass
class FinancialRubric:
    """Rubric for grading financial RAG responses.

    Adapted from Projects/ARO/src/optimizer/schema.py.
    """

    name: str = "default"
    dimensions: list[str] = field(
        default_factory=lambda: [
            "factual_accuracy",
            "quantitative_precision",
            "source_attribution",
            "analytical_depth",
            "completeness",
        ]
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "factual_accuracy": 0.30,
            "quantitative_precision": 0.25,
            "source_attribution": 0.20,
            "analytical_depth": 0.15,
            "completeness": 0.10,
        }
    )

    def weighted_score(self, scores: dict[str, float]) -> float:
        """Calculate weighted score from dimension scores (0-1 scale)."""
        total = 0.0
        for dim, weight in self.weights.items():
            total += scores.get(dim, 0.0) * weight
        return total
