"""Answer quality grading using rubrics.

Adapted from multi-agent-investment-committee (evals/grader.py).
Supports both automated heuristic grading and LLM-based grading.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from rag.evaluation.schemas import EvalResult, EvalScenario, LikertScore

logger = logging.getLogger(__name__)


class AnswerGrader:
    """Grade RAG answers against expected outputs using heuristic scoring."""

    def grade(
        self,
        scenario: EvalScenario,
        generated_answer: str,
        retrieval_count: int = 0,
        citation_count: int = 0,
    ) -> EvalResult:
        """Grade a generated answer against a scenario.

        Uses heuristic scoring based on:
        - Text similarity to expected answer (if provided)
        - Citation presence and count
        - Answer length and completeness signals

        Args:
            scenario: The eval scenario with expected answer.
            generated_answer: The RAG-generated answer.
            retrieval_count: Number of documents retrieved.
            citation_count: Number of citations in the answer.

        Returns:
            An ``EvalResult`` with quality scores.
        """
        relevance = self._score_relevance(scenario.question, generated_answer)
        accuracy = self._score_accuracy(scenario.expected_answer, generated_answer)
        completeness = self._score_completeness(generated_answer)
        citation_quality = self._score_citations(generated_answer, citation_count)

        return EvalResult(
            scenario_id=scenario.id,
            question=scenario.question,
            generated_answer=generated_answer,
            retrieval_count=retrieval_count,
            citation_count=citation_count,
            relevance=relevance,
            accuracy=accuracy,
            completeness=completeness,
            citation_quality=citation_quality,
        )

    @staticmethod
    def _score_relevance(question: str, answer: str) -> LikertScore:
        """Score how relevant the answer is to the question."""
        if not answer.strip():
            return LikertScore.VERY_POOR

        # Check for question keyword overlap
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words) / max(len(q_words), 1)

        if overlap > 0.4:
            return LikertScore.GOOD
        elif overlap > 0.2:
            return LikertScore.ADEQUATE
        else:
            return LikertScore.POOR

    @staticmethod
    def _score_accuracy(expected: str | None, generated: str) -> LikertScore:
        """Score accuracy by comparing to expected answer."""
        if expected is None:
            return LikertScore.ADEQUATE  # Can't grade without expected

        similarity = SequenceMatcher(None, expected.lower(), generated.lower()).ratio()

        if similarity > 0.7:
            return LikertScore.EXCELLENT
        elif similarity > 0.5:
            return LikertScore.GOOD
        elif similarity > 0.3:
            return LikertScore.ADEQUATE
        elif similarity > 0.15:
            return LikertScore.POOR
        else:
            return LikertScore.VERY_POOR

    @staticmethod
    def _score_completeness(answer: str) -> LikertScore:
        """Score answer completeness based on length and structure."""
        word_count = len(answer.split())

        if word_count < 10:
            return LikertScore.VERY_POOR
        elif word_count < 30:
            return LikertScore.POOR
        elif word_count < 80:
            return LikertScore.ADEQUATE
        elif word_count < 200:
            return LikertScore.GOOD
        else:
            return LikertScore.EXCELLENT

    @staticmethod
    def _score_citations(answer: str, citation_count: int) -> LikertScore:
        """Score citation quality."""
        # Count citation references in the text
        refs = len(re.findall(r"\[\d+\]", answer))

        if refs >= 3:
            return LikertScore.EXCELLENT
        elif refs >= 2:
            return LikertScore.GOOD
        elif refs >= 1:
            return LikertScore.ADEQUATE
        else:
            return LikertScore.POOR
