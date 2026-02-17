"""Boilerplate and disclaimer removal for financial documents.

Merged from:
- redflag_ex1_analyst (boilerplate_filter.py) — equity research disclaimers
- investment-workflow-evals (studio/document.py) — SEC filing boilerplate

Two-pass approach:
1. Section-level: remove from boilerplate header to next substantive header
2. Paragraph-level: regex match against individual paragraphs
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BoilerplateFilterConfig:
    """Toggle individual filter categories."""

    enabled: bool = True
    strip_disclaimers: bool = True
    strip_certifications: bool = True
    strip_distribution_notices: bool = True
    strip_regulatory_notices: bool = True
    strip_confidentiality_notices: bool = True
    strip_copyright_notices: bool = True
    strip_forward_looking: bool = True
    custom_patterns: list[str] = field(default_factory=list)
    protected_keywords: list[str] = field(
        default_factory=lambda: [
            "insider", "not public", "13f", "material nonpublic",
        ]
    )


@dataclass
class FilterResult:
    """Result of boilerplate filtering."""

    text: str
    chars_removed: int = 0
    sections_removed: int = 0
    paragraphs_removed: int = 0


# ---------------------------------------------------------------------------
# Section-level boilerplate headers
# ---------------------------------------------------------------------------

_SECTION_BOILERPLATE_RE = re.compile(
    r"(?i)^(?:"
    r"important\s+disclosures?"
    r"|disclosure\s+appendix"
    r"|regulatory\s+disclosures?"
    r"|analyst\s+certifications?"
    r"|general\s+disclosures?"
    r"|required\s+disclosures?"
    r"|legal\s+disclosures?"
    r"|distribution\s+of\s+ratings"
    r"|valuation\s+methodology\s+(?:and\s+)?risk"
    r")"
)

# ---------------------------------------------------------------------------
# Paragraph-level discard patterns
# ---------------------------------------------------------------------------

_PARA_DISCARD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)not\s+an?\s+offer\s+to\s+sell"),
    re.compile(r"(?i)past\s+performance\s+(?:is\s+)?not\s+(?:necessarily\s+)?(?:indicative|a\s+guarantee)"),
    re.compile(r"(?i)this\s+(?:report|document|research)\s+(?:is|was)\s+prepared\s+(?:by|for)"),
    re.compile(r"(?i)(?:all|any)\s+(?:rights?\s+)?reserved"),
    re.compile(r"(?i)forward[- ]looking\s+statements?"),
    re.compile(r"(?i)safe\s+harbor\s+(?:statement|provision)"),
    re.compile(r"(?i)(?:sox|sarbanes[- ]oxley)\s+(?:section|certification)"),
    re.compile(r"(?i)xbrl\s+(?:instance|taxonomy|viewer)"),
    re.compile(r"(?i)edgar\s+(?:filing|header|submission)"),
    re.compile(r"(?i)pursuant\s+to\s+(?:section|rule)\s+\d"),
    re.compile(r"(?i)(?:the\s+)?securities\s+and\s+exchange\s+commission\s+has\s+not"),
    re.compile(r"(?i)this\s+communication\s+is\s+(?:not\s+)?(?:intended|directed)"),
]


class BoilerplateFilter:
    """Remove boilerplate text from financial documents."""

    def __init__(self, config: BoilerplateFilterConfig | None = None):
        self.config = config or BoilerplateFilterConfig()
        self._extra_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.custom_patterns
        ]

    def filter(self, text: str) -> FilterResult:
        """Apply two-pass boilerplate removal.

        Returns a ``FilterResult`` with cleaned text and removal stats.
        """
        if not self.config.enabled:
            return FilterResult(text=text)

        original_len = len(text)

        # Pass 1: section-level removal
        text, sections_removed = self._strip_sections(text)

        # Pass 2: paragraph-level removal
        text, paragraphs_removed = self._strip_paragraphs(text)

        chars_removed = original_len - len(text)
        if chars_removed > 0:
            pct = chars_removed / original_len * 100 if original_len else 0
            logger.info(
                "Boilerplate filter removed %d chars (%.1f%%), "
                "%d sections, %d paragraphs",
                chars_removed, pct, sections_removed, paragraphs_removed,
            )

        return FilterResult(
            text=text,
            chars_removed=chars_removed,
            sections_removed=sections_removed,
            paragraphs_removed=paragraphs_removed,
        )

    # ------------------------------------------------------------------
    # Pass 1 — section-level
    # ------------------------------------------------------------------

    def _strip_sections(self, text: str) -> tuple[str, int]:
        lines = text.split("\n")
        output: list[str] = []
        skipping = False
        removed = 0

        for line in lines:
            stripped = line.strip()
            if _SECTION_BOILERPLATE_RE.match(stripped):
                skipping = True
                removed += 1
                continue

            if (
                skipping
                and stripped
                and not stripped[0].isspace()
                and len(stripped) < 120
                and stripped[0].isupper()
            ):
                # Hit a new non-indented header — stop skipping
                skipping = False

            if skipping:
                continue

            output.append(line)

        return "\n".join(output), removed

    # ------------------------------------------------------------------
    # Pass 2 — paragraph-level
    # ------------------------------------------------------------------

    def _strip_paragraphs(self, text: str) -> tuple[str, int]:
        paragraphs = re.split(r"\n{2,}", text)
        kept: list[str] = []
        removed = 0
        all_patterns = _PARA_DISCARD_PATTERNS + self._extra_patterns

        for para in paragraphs:
            if self._is_protected(para):
                kept.append(para)
                continue

            if any(p.search(para) for p in all_patterns):
                removed += 1
                continue

            kept.append(para)

        return "\n\n".join(kept), removed

    def _is_protected(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in self.config.protected_keywords)
