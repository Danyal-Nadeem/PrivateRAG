"""Hybrid NER filter combining rule-based regex with token classification.

Arbitration Rules:
- Regex WINS for deterministic, pattern-based entities:
    EMAIL, PHONE, EMP_ID, SALARY
- Model WINS for contextual, language-based entities:
    PERSON, ADDRESS

Any overlapping conflicts are resolved in favor of the designated winner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

from src.ner.regex_filter import extract_regex_entities
from src.ner.infer import extract_model_entities

REGEX_PRIORITY_LABELS = frozenset({"EMAIL", "PHONE", "EMP_ID", "SALARY"})
MODEL_PRIORITY_LABELS = frozenset({"PERSON", "ADDRESS"})


def _spans_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if span a and span b share any character indices."""
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def extract_hybrid_entities(
    text: str,
    model_dir: Path | str | None = None,
    use_model: bool = True,
) -> List[Dict[str, Any]]:
    """Extract entities from *text* using the hybrid (Regex + Model) strategy.

    Parameters
    ----------
    text:
        Raw input string.
    model_dir:
        Optional path to fine-tuned model checkpoint directory.
    use_model:
        Whether to run model inference (defaults to True; set False for regex-only).

    Returns
    -------
    List[Dict[str, Any]]
        Resolved non-overlapping entity spans sorted by start offset.
    """
    regex_spans = extract_regex_entities(text)
    model_spans = extract_model_entities(text, model_dir=model_dir) if use_model else []

    accepted_spans: List[Dict[str, Any]] = []

    # 1. Regex spans for pattern-based entities are accepted first
    for r_span in regex_spans:
        if r_span["label"] in REGEX_PRIORITY_LABELS:
            accepted_spans.append(r_span)

    # 2. Add model spans for contextual entities (PERSON, ADDRESS) if they don't
    # conflict with accepted regex pattern spans
    for m_span in model_spans:
        if m_span["label"] in MODEL_PRIORITY_LABELS:
            # Check overlap against all accepted regex spans
            overlaps = any(_spans_overlap(m_span, acc) for acc in accepted_spans)
            if not overlaps:
                accepted_spans.append(m_span)

    # 3. If model detected pattern entities that regex somehow missed, add them if non-conflicting
    for m_span in model_spans:
        if m_span["label"] in REGEX_PRIORITY_LABELS:
            overlaps = any(_spans_overlap(m_span, acc) for acc in accepted_spans)
            if not overlaps:
                accepted_spans.append(m_span)

    # 4. Sort and do a final non-overlapping pass
    accepted_spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

    final_spans: List[Dict[str, Any]] = []
    last_end = -1
    for s in accepted_spans:
        if s["start"] >= last_end:
            final_spans.append(s)
            last_end = s["end"]

    return final_spans
