"""LLM provider factory. Select provider via LLM_PROVIDER env var."""
from __future__ import annotations

import os


def get_llm_provider():
    """Return the configured LLM provider instance.

    Defaults to 'extractive' so the system works with no API keys.
    """
    provider = os.getenv("LLM_PROVIDER", "extractive")
    if provider == "extractive":
        from src.llm.extractive import ExtractiveProvider
        return ExtractiveProvider()
    elif provider == "gemini":
        from src.llm.gemini import GeminiProvider
        return GeminiProvider()
    elif provider == "local_hf":
        from src.llm.local_hf import LocalHFProvider
        return LocalHFProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Choose: extractive, gemini, local_hf")
