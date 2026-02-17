"""Ollama LLM provider — local-first, no API keys.

Supports DeepSeek-R1, Llama, Mistral, and any model available via Ollama.
"""

from __future__ import annotations

import logging

import httpx

from rag.llm.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-r1:32b"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLMProvider(LLMProvider):
    """Generate responses via a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")
