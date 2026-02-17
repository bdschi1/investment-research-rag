"""OpenAI embedding provider — text-embedding-3-small/large.

Requires ``openai`` extra and an API key via ``OPENAI_API_KEY`` env var.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"

_DIMENSION_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

BATCH_SIZE = 2048  # OpenAI max batch size


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embed text via the OpenAI Embeddings API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        dimensions: int | None = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package required: pip install investment-research-rag[openai]"
            ) from exc

        self.model = model
        self._dimensions = dimensions or _DIMENSION_MAP.get(model, 1536)
        self._client: Any = openai.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            # Sort by index to guarantee order
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return resp.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimensions
