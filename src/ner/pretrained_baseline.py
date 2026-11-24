"""Pretrained NER baseline using dslim/bert-base-NER with label mapping.

CoNLL-2003 labels -> PrivateRAG labels:
- B-PER / I-PER -> PERSON
- Other labels (LOC, ORG, MISC) are outside our private entity schema.

Returns character-offset spans:
    [{"start": int, "end": int, "label": str, "text": str}]
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Any

_MODEL_NAME = "dslim/bert-base-NER"
_pipeline = None
_pipeline_failed = False
_pipeline_failed_at: float | None = None

# Retry window for transient failures (network, download) — a permanent
# offline mode (HF_HUB_OFFLINE / TESTING) still disables the baseline for good.
_RETRY_COOLDOWN_SECONDS = 60.0
_now = time.monotonic


def reset_baseline_cache() -> None:
    """Forget cached pipeline state so the baseline is re-attempted."""
    global _pipeline, _pipeline_failed, _pipeline_failed_at
    _pipeline = None
    _pipeline_failed = False
    _pipeline_failed_at = None


def _get_pipeline():
    global _pipeline, _pipeline_failed, _pipeline_failed_at
    if _pipeline_failed:
        return None
    if _pipeline is None:
        is_offline = (
            os.getenv("HF_HUB_OFFLINE") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE") == "1"
            or os.getenv("TESTING") == "1"
        )
        if is_offline:
            _pipeline_failed = True
            return None

        # Bound retries of transient failures to one attempt per cooldown
        # window, so an eval loop doesn't hammer the network on every record.
        if (
            _pipeline_failed_at is not None
            and _now() - _pipeline_failed_at < _RETRY_COOLDOWN_SECONDS
        ):
            return None

        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "token-classification",
                model=_MODEL_NAME,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
        except Exception:
            _pipeline_failed_at = _now()
            return None
    return _pipeline


def pretrained_available() -> bool:
    """Return True if the pretrained baseline pipeline can be loaded.

    Used by evaluation harnesses to avoid silently scoring an unavailable
    baseline as zero predictions.
    """
    return _get_pipeline() is not None


def extract_pretrained_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities using dslim/bert-base-NER mapped to our label schema.

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    List[Dict[str, Any]]
        Detected spans with label 'PERSON'.
    """
    pipe = _get_pipeline()
    if pipe is None or not text.strip():
        return []

    try:
        raw_results = pipe(text)
    except Exception:
        return []

    spans: List[Dict[str, Any]] = []
    for r in raw_results:
        entity_group = r.get("entity_group", "")
        # Map PER -> PERSON
        if entity_group == "PER":
            spans.append({
                "start": int(r["start"]),
                "end": int(r["end"]),
                "label": "PERSON",
                "text": text[int(r["start"]):int(r["end"])].strip(),
            })

    # Sort spans by start offset
    spans.sort(key=lambda s: s["start"])
    return spans
