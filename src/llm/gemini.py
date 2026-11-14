"""Gemini provider stub.

Requirements
------------
* ``GEMINI_API_KEY``  — from .env, never logged or printed.
* ``GEMINI_MODEL``    — from .env, required (no default).

The ``google-generativeai`` package is NOT installed in this project.
Importing this module is safe; calling ``generate()`` without the package
raises a clear RuntimeError.
"""
from __future__ import annotations

import os

from src.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Gemini API."""

    def __init__(self) -> None:
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env."
            )
        self._model = os.getenv("GEMINI_MODEL")
        if not self._model:
            raise RuntimeError(
                "GEMINI_MODEL is not set. Add it to .env (e.g. GEMINI_MODEL=gemini-2.0-flash)."
            )
        # Lazy import — raises RuntimeError if package is absent
        try:
            import google.generativeai as _genai  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "gemini provider requires the google-generativeai package. "
                "Install it with: pip install google-generativeai"
            )

    def generate(self, prompt: str, context_chunks: list[str] | None = None) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        full_prompt = prompt
        if context_chunks:
            context = "\n\n".join(context_chunks)
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
        response = model.generate_content(full_prompt)
        return response.text
