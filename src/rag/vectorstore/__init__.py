"""Vector store backends — FAISS (local) and Qdrant (production)."""

from rag.vectorstore.base import VectorStore
from rag.vectorstore.factory import available_stores, get_vector_store
from rag.vectorstore.schemas import MetadataFilter, SearchResult, VectorRecord

__all__ = [
    "MetadataFilter",
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "available_stores",
    "get_vector_store",
]
