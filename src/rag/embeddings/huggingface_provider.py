"""HuggingFace/sentence-transformers embedding provider.

Runs locally via ``sentence-transformers``. Requires the ``huggingface`` extra.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embed text locally using sentence-transformers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required: "
                "pip install investment-research-rag[huggingface]"
            ) from exc

        self._model_name = model
        self._model: Any = SentenceTransformer(model, device=device)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded HF model %s (dim=%d)", model, self._dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dim
