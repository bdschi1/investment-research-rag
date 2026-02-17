"""Smart page scoring for large documents.

Ported from multi-agent-investment-committee (tools/doc_chunker.py).
Assigns importance scores to pages based on content heuristics, allowing
budget-constrained page selection for embedding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# High-value header patterns
# ---------------------------------------------------------------------------

_HIGH_VALUE_HEADERS = re.compile(
    r"(?i)(?:"
    r"executive\s+summary"
    r"|investment\s+(?:thesis|summary|conclusion)"
    r"|key\s+(?:findings|takeaways|drivers|risks)"
    r"|financial\s+(?:summary|highlights|overview)"
    r"|valuation"
    r"|price\s+target"
    r"|recommendation"
    r"|conclusion"
    r"|risk\s+factors?"
    r"|catalysts?"
    r"|earnings"
    r"|revenue"
    r"|guidance"
    r"|outlook"
    r"|management\s+discussion"
    r"|md\s*&\s*a"
    r")"
)

_TABLE_INDICATOR = re.compile(r"(?:\|.*\||\t.*\t|^\s*\d+[\.,]\d+\s*$)", re.MULTILINE)


@dataclass
class ScoredPage:
    """A page with an importance score."""

    page_index: int
    text: str
    score: float


def score_page(text: str, page_index: int, total_pages: int) -> float:
    """Score a single page for importance.

    Scoring rules:
        +3.0  High-value headers (executive summary, thesis, etc.)
        +2.0  Table-like content detected
        +1.5  High digit density (>5% — quantitative content)
        +1.0  First 2 pages or last page (always valuable)
        -2.0  Very short pages (<200 chars — covers, disclaimers)
    """
    score = 0.0

    # High-value headers
    if _HIGH_VALUE_HEADERS.search(text):
        score += 3.0

    # Tables
    if _TABLE_INDICATOR.search(text):
        score += 2.0

    # Digit density
    digits = sum(1 for c in text if c.isdigit())
    total = len(text) or 1
    if digits / total > 0.05:
        score += 1.5

    # Position bonuses
    if page_index < 2 or page_index == total_pages - 1:
        score += 1.0

    # Penalty for very short pages
    if len(text.strip()) < 200:
        score -= 2.0

    return score


def select_pages(
    page_texts: list[str],
    max_pages: int | None = None,
) -> list[ScoredPage]:
    """Score and select the most important pages.

    Args:
        page_texts: Per-page text content.
        max_pages: Maximum pages to return. None = return all, scored.

    Returns:
        List of ``ScoredPage`` sorted by original page order.
    """
    total = len(page_texts)
    scored = [
        ScoredPage(
            page_index=i,
            text=text,
            score=score_page(text, i, total),
        )
        for i, text in enumerate(page_texts)
    ]

    if max_pages is not None and max_pages < total:
        # Always include first 2 + last page, then fill by score
        must_include = set()
        if total > 0:
            must_include.add(0)
        if total > 1:
            must_include.add(1)
        if total > 2:
            must_include.add(total - 1)

        remaining = [s for s in scored if s.page_index not in must_include]
        remaining.sort(key=lambda s: s.score, reverse=True)

        budget = max_pages - len(must_include)
        selected_indices = must_include | {
            s.page_index for s in remaining[:max(0, budget)]
        }

        scored = [s for s in scored if s.page_index in selected_indices]

    # Return in original page order
    scored.sort(key=lambda s: s.page_index)

    logger.info(
        "Page scoring: %d/%d pages selected (scores: %.1f to %.1f)",
        len(scored), total,
        min(s.score for s in scored) if scored else 0,
        max(s.score for s in scored) if scored else 0,
    )
    return scored
