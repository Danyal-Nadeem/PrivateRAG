"""Extractive provider — returns the top retrieved chunk as the answer.

No model loading, no API keys, no network calls.
This is the default provider (LLM_PROVIDER=extractive).
"""
from __future__ import annotations

from src.llm.base import LLMProvider


class ExtractiveProvider(LLMProvider):
    """Zero-dependency provider that extracts the best chunk as the answer."""

    def generate(self, prompt: str, context_chunks: list[str] | None = None) -> str:
        """Return the first context chunk, or a fallback message.

        Parameters
        ----------
        prompt:
            Ignored by this provider.
        context_chunks:
            List of retrieved chunk texts. The first one is returned.

        Returns
        -------
        str
            The top retrieved chunk, or a "no results" message.
        """
        if context_chunks:
            return context_chunks[0].strip()
        return "No relevant information found for your query."
