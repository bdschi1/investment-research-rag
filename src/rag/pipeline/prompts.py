"""Financial-domain prompt templates for RAG pipeline.

All prompts are designed for institutional investment research context.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a senior investment research analyst. Answer questions using ONLY \
the provided context documents. If the context does not contain enough \
information to answer the question, say so explicitly.

Rules:
1. Cite sources using [1], [2], etc. corresponding to the numbered context \
documents below.
2. Be precise with financial figures — do not round unless the source rounds.
3. Distinguish between facts stated in the documents and your own analysis.
4. If multiple sources conflict, note the discrepancy.
5. Use professional investment language appropriate for institutional audiences.
"""

RAG_QUERY_TEMPLATE = """\
Context Documents:
{context}

Question: {question}

Answer the question using only the context documents above. Cite sources \
using [1], [2], etc.
"""

# ---------------------------------------------------------------------------
# Specialized prompts
# ---------------------------------------------------------------------------

SEC_ANALYSIS_PROMPT = """\
You are analyzing an SEC filing. Focus on material changes, risk factors, \
and financial trends. Use specific figures from the filing and note any \
year-over-year changes.
"""

EARNINGS_ANALYSIS_PROMPT = """\
You are analyzing an earnings call transcript. Focus on management guidance, \
key metrics discussed, analyst concerns, and any forward-looking statements. \
Attribute statements to specific speakers when possible.
"""

VALUATION_PROMPT = """\
You are a valuation analyst. Focus on quantitative metrics: revenue growth, \
margins, multiples, DCF assumptions, and comparable company analysis. Be \
precise with numbers and note any assumptions or limitations.
"""


def format_context(texts: list[str], sources: list[str] | None = None) -> str:
    """Format context documents for insertion into the prompt.

    Args:
        texts: The context document texts.
        sources: Optional source labels (filenames, tickers, etc.).

    Returns:
        Formatted context string with numbered documents.
    """
    parts = []
    for i, text in enumerate(texts, 1):
        label = f"[{i}]"
        if sources and i - 1 < len(sources):
            label += f" ({sources[i - 1]})"
        parts.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(parts)


def build_rag_prompt(
    question: str,
    context_texts: list[str],
    sources: list[str] | None = None,
) -> str:
    """Build a complete RAG prompt with context and question.

    Args:
        question: The user's question.
        context_texts: Retrieved context documents.
        sources: Optional source labels.

    Returns:
        The formatted prompt string.
    """
    context = format_context(context_texts, sources)
    return RAG_QUERY_TEMPLATE.format(context=context, question=question)
