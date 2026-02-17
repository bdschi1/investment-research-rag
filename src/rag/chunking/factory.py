"""Chunker factory — auto-detect document type and dispatch to the right chunker.

Follows the provider factory pattern from multi-agent-investment-committee
(tools/data_providers/factory.py).
"""

from __future__ import annotations

import importlib
import logging

from rag.chunking.base import BaseChunker
from rag.documents.schemas import DocType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunker registry
#
# Each entry: (doc_type, module_path, class_name)
# ---------------------------------------------------------------------------

_CHUNKER_REGISTRY: list[tuple[DocType, str, str]] = [
    (DocType.RESEARCH_REPORT, "rag.chunking.research_chunker", "ResearchChunker"),
    (DocType.SEC_FILING, "rag.chunking.sec_chunker", "SecChunker"),
    (DocType.EARNINGS_TRANSCRIPT, "rag.chunking.transcript_chunker", "TranscriptChunker"),
    (DocType.FINANCIAL_MODEL, "rag.chunking.excel_chunker", "ExcelChunker"),
]

# Singleton cache
_chunker_cache: dict[DocType, BaseChunker] = {}


def get_chunker(doc_type: DocType = DocType.OTHER, **kwargs) -> BaseChunker:
    """Get a chunker for the given document type.

    Falls back to ``ResearchChunker`` for unknown types (it handles
    generic text well).
    """
    if not kwargs and doc_type in _chunker_cache:
        return _chunker_cache[doc_type]

    for registered_type, module_path, cls_name in _CHUNKER_REGISTRY:
        if registered_type == doc_type:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            instance = cls(**kwargs)
            if not kwargs:
                _chunker_cache[doc_type] = instance
            return instance

    # Default: research chunker works for generic text
    logger.info("No specific chunker for %s, using ResearchChunker", doc_type)
    from rag.chunking.research_chunker import ResearchChunker

    instance = ResearchChunker(**kwargs)
    if not kwargs:
        _chunker_cache[doc_type] = instance
    return instance


def available_chunkers() -> list[str]:
    """Return names of registered chunkers."""
    return [dt.value for dt, _, _ in _CHUNKER_REGISTRY]


def clear_cache() -> None:
    """Clear singleton cache (for testing)."""
    _chunker_cache.clear()
