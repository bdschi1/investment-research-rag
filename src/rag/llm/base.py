"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for LLM response generation."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            Generated text response.
        """

    @classmethod
    def provider_name(cls) -> str:
        """Return human-readable provider name."""
        return cls.__name__
