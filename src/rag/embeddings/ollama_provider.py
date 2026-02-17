"""Ollama embedding provider — local-first, no API keys needed.

Uses the Ollama REST API (http://localhost:11434) with models like
``nomic-embed-text``, ``mxbai-embed-large``, etc.
"""

from __future__ import annotations

import logging

import httpx

from rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_DIM = 768


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embed text via a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimension: int = DEFAULT_DIM,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = dimension
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts by calling Ollama one at a time.

        Ollama's /api/embed endpoint supports batch input since v0.5+.
        Falls back to sequential calls for older versions.
        """
        if not texts:
            return []

        # Try batch first (Ollama v0.5+)
        try:
            resp = self._client.post(
                "/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"]
        except (httpx.HTTPError, KeyError):
            pass

        # Fallback: sequential
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_single(text))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self._embed_single(query)

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_single(self, text: str) -> list[float]:
        resp = self._client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
