"""Pretrained NER baseline using dslim/bert-base-NER with label mapping.

CoNLL-2003 labels -> PrivateRAG labels:
- B-PER / I-PER -> PERSON
- Other labels (LOC, ORG, MISC) are outside our private entity schema.

Returns character-offset spans:
    [{"start": int, "end": int, "label": str, "text": str}]
"""
from __future__ import annotations

import os
from typing import Dict, List, Any

_MODEL_NAME = "dslim/bert-base-NER"
_pipeline = None
_pipeline_failed = False


def _get_pipeline():
    global _pipeline, _pipeline_failed
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

        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "token-classification",
                model=_MODEL_NAME,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
        except Exception:
            _pipeline_failed = True
            return None
    return _pipeline


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
