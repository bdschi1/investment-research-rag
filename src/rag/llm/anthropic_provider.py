"""Anthropic Claude LLM provider.

Requires the ``anthropic`` extra and ``ANTHROPIC_API_KEY`` env var.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.llm.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicLLMProvider(LLMProvider):
    """Generate responses via the Anthropic API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required: pip install investment-research-rag[anthropic]"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text
