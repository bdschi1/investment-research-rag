"""Citation extraction and source mapping.

Parses [1], [2], etc. from LLM output and maps them back to source documents.
Adapted from Projects/z_oldProjects/doc-chunker/rag.py citation logic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.pipeline.schemas import Citation
from rag.vectorstore.schemas import SearchResult

# Matches [1], [2], [3,4], [1-3], etc.
_CITATION_RE = re.compile(r"\[(\d+(?:[,\-]\d+)*)\]")


def extract_citations(
    answer: str,
    search_results: Sequence[SearchResult],
) -> list[Citation]:
    """Extract citation references from an LLM answer and map to sources.

    Args:
        answer: The LLM-generated answer text.
        search_results: The search results that were provided as context.

    Returns:
        List of ``Citation`` objects with source information.
    """
    # Find all citation numbers in the answer
    cited_indices: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        group = match.group(1)
        # Handle ranges like [1-3] and lists like [1,2]
        for part in group.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                for i in range(int(start), int(end) + 1):
                    cited_indices.add(i)
            else:
                cited_indices.add(int(part))

    # Map citation numbers to search results (1-indexed)
    citations: list[Citation] = []
    for idx in sorted(cited_indices):
        result_idx = idx - 1  # Convert to 0-indexed
        if 0 <= result_idx < len(search_results):
            result = search_results[result_idx]
            # Truncate text for citation display
            snippet = result.text[:200] + "..." if len(result.text) > 200 else result.text
            citations.append(Citation(
                index=idx,
                text=snippet,
                source=result.metadata.source_filename or "unknown",
                section=result.metadata.section_name,
                ticker=result.metadata.ticker,
                score=result.score,
            ))

    return citations


def format_citations(citations: list[Citation]) -> str:
    """Format citations for display.

    Returns a markdown-formatted citation block.
    """
    if not citations:
        return ""

    lines = ["\n---\n**Sources:**"]
    for c in citations:
        parts = [f"[{c.index}]"]
        if c.ticker:
            parts.append(c.ticker)
        if c.source and c.source != "unknown":
            parts.append(c.source)
        if c.section:
            parts.append(f"({c.section})")
        lines.append(f"- {' | '.join(parts)}")

    return "\n".join(lines)
