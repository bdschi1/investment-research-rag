"""OpenAI LLM provider — GPT-4, GPT-4o, and OpenAI-compatible endpoints.

Requires the ``openai`` extra and ``OPENAI_API_KEY`` env var.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.llm.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


class OpenAILLMProvider(LLMProvider):
    """Generate responses via the OpenAI Chat API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package required: pip install investment-research-rag[openai]"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client: Any = openai.OpenAI(**kwargs)

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
