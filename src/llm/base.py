"""Abstract base class for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for all LLM back-ends."""

    @abstractmethod
    def generate(self, prompt: str, context_chunks: list[str] | None = None) -> str:
        """Generate a response given a prompt and optional retrieved chunks.

        Parameters
        ----------
        prompt:
            The user question (after any RAG augmentation by the caller).
        context_chunks:
            Retrieved chunk texts for the current query.  Some providers
            (e.g. extractive) use these directly; others (e.g. Gemini) may
            ignore them and rely on the prompt containing the context.

        Returns
        -------
        str
            The generated answer text.
        """
