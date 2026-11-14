"""Singleton sentence-transformer embedder.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80 MB).
The model is loaded once at module level and reused across calls.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # local_files_only avoids network round-trips to HuggingFace Hub
        # when the model is already cached; falls back to online if not cached.
        import os
        try:
            _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode *texts* and return an (N, 384) float32 numpy array.

    Parameters
    ----------
    texts:
        List of strings to embed.
    batch_size:
        Mini-batch size passed to the underlying model.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), 384)``, dtype ``float32``.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(vectors, dtype=np.float32)
