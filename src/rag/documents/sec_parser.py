"""SEC filing section detection for 10-K, 10-Q, and 8-K documents.

Detects ITEM and PART boundaries via regex pattern matching against
extracted page text. Adapted from investment-workflow-evals
(studio/document.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section patterns
# ---------------------------------------------------------------------------

# Matches "ITEM 1.", "ITEM 1A.", "Item 7 -", "ITEM 7. ", etc.
_ITEM_RE = re.compile(
    r"(?i)^(?:ITEM)\s+(\d+[A-Z]?)[\.\s\-\u2013\u2014]",
    re.MULTILINE,
)

# Matches "PART I", "PART II", "Part IV", etc.
_PART_RE = re.compile(
    r"(?i)^(?:PART)\s+(I{1,4}|IV|V|VI{0,3})[\.\s\-\u2013\u2014]?",
    re.MULTILINE,
)

# Common item titles for labeling
_ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}


@dataclass(frozen=True)
class Section:
    """A detected section within an SEC filing."""

    title: str
    item_number: str
    start_page: int
    end_page: int
    start_char: int
    end_char: int


@dataclass
class ParsedFiling:
    """Result of parsing an SEC filing into sections."""

    total_pages: int
    total_chars: int
    sections: list[Section]
    has_sections: bool


def parse_sec_filing(page_texts: list[str]) -> ParsedFiling:
    """Detect ITEM sections in an SEC filing from per-page text.

    Args:
        page_texts: List of strings, one per page.

    Returns:
        A ``ParsedFiling`` with detected sections.
    """
    if not page_texts:
        return ParsedFiling(
            total_pages=0, total_chars=0, sections=[], has_sections=False
        )

    full_text = "\n\n".join(page_texts)
    total_chars = len(full_text)

    # Build page boundary offsets
    page_offsets: list[int] = []
    offset = 0
    for pt in page_texts:
        page_offsets.append(offset)
        offset += len(pt) + 2  # +2 for the \n\n separator

    # Find all ITEM matches
    matches: list[tuple[int, str]] = []
    for m in _ITEM_RE.finditer(full_text):
        item_num = m.group(1).upper()
        matches.append((m.start(), item_num))

    if not matches:
        return ParsedFiling(
            total_pages=len(page_texts),
            total_chars=total_chars,
            sections=[],
            has_sections=False,
        )

    # Build sections from consecutive matches
    sections: list[Section] = []
    for i, (start_char, item_num) in enumerate(matches):
        end_char = matches[i + 1][0] if i + 1 < len(matches) else total_chars
        start_page = _char_to_page(start_char, page_offsets)
        end_page = _char_to_page(end_char - 1, page_offsets)

        label = _ITEM_TITLES.get(item_num, "")
        title = f"Item {item_num}" + (f". {label}" if label else "")

        sections.append(Section(
            title=title,
            item_number=item_num,
            start_page=start_page,
            end_page=end_page,
            start_char=start_char,
            end_char=end_char,
        ))

    logger.info(
        "Detected %d sections in SEC filing (%d pages)",
        len(sections), len(page_texts),
    )

    return ParsedFiling(
        total_pages=len(page_texts),
        total_chars=total_chars,
        sections=sections,
        has_sections=True,
    )


def _char_to_page(char_offset: int, page_offsets: list[int]) -> int:
    """Map a character offset to a 0-based page number."""
    for i in range(len(page_offsets) - 1, -1, -1):
        if char_offset >= page_offsets[i]:
            return i
    return 0
