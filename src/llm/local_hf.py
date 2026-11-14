"""Local HuggingFace provider stub.

Uses a small seq2seq model (google/flan-t5-small by default).
The ``transformers`` package IS installed as a transitive dependency of
sentence-transformers, so the import will succeed.  However, this provider
downloads a model on first use (~300 MB for flan-t5-small) and therefore
should not be called during unit tests.

Set LLM_PROVIDER=local_hf to use this provider.
"""
from __future__ import annotations

import os

from src.llm.base import LLMProvider

_HF_MODEL_DEFAULT = "google/flan-t5-small"


class LocalHFProvider(LLMProvider):
    """LLM provider backed by a small local HuggingFace model."""

    def __init__(self) -> None:
        self._model_name = os.getenv("HF_MODEL", _HF_MODEL_DEFAULT)
        try:
            from transformers import pipeline  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "local_hf provider requires the transformers package. "
                "Install it with: pip install transformers"
            )
        self._pipeline = None  # Lazy — load on first generate() call

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text2text-generation",
                model=self._model_name,
                max_new_tokens=256,
            )
        return self._pipeline

    def generate(self, prompt: str, context_chunks: list[str] | None = None) -> str:
        pipe = self._get_pipeline()
        full_prompt = prompt
        if context_chunks:
            context = "\n\n".join(context_chunks[:3])
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        result = pipe(full_prompt)
        return result[0]["generated_text"].strip()
