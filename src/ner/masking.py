"""Entity masking engine that redacts sensitive spans with standardized tokens.

Mask tokens:
    [PERSON_REDACTED], [EMAIL_REDACTED], [PHONE_REDACTED],
    [EMP_ID_REDACTED], [ADDRESS_REDACTED], [SALARY_REDACTED]

Replacements are executed strictly right-to-left (descending by start offset)
so that preceding character offsets remain invariant throughout replacement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

from src.ner.regex_filter import extract_regex_entities
from src.ner.hybrid_filter import extract_hybrid_entities

MASK_TOKEN_MAP: dict[str, str] = {
    "PERSON": "[PERSON_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "EMP_ID": "[EMP_ID_REDACTED]",
    "ADDRESS": "[ADDRESS_REDACTED]",
    "SALARY": "[SALARY_REDACTED]",
}


def mask_entities(text: str, spans: List[Dict[str, Any]]) -> str:
    """Replace *spans* in *text* with mask tokens, executing right-to-left.

    Parameters
    ----------
    text:
        Raw input string.
    spans:
        List of dicts with keys: ``start``, ``end``, ``label``, ``text``.

    Returns
    -------
    str
        Sanitized string with sensitive spans replaced by mask tokens.
    """
    if not spans or not text:
        return text

    # Filter valid spans and sort strictly descending by start offset
    valid_spans = [
        s for s in spans
        if 0 <= s["start"] < s["end"] <= len(text)
    ]
    # If overlapping, deduplicate: sort by start descending
    sorted_spans = sorted(valid_spans, key=lambda s: (s["start"], s["end"]), reverse=True)

    # Right-to-left non-overlapping replacement
    masked = text
    last_start = len(text) + 1

    for s in sorted_spans:
        # Prevent overlapping corruption
        if s["end"] <= last_start:
            label = s.get("label", "ENTITY")
            token = MASK_TOKEN_MAP.get(label, f"[{label}_REDACTED]")
            masked = masked[:s["start"]] + token + masked[s["end"]:]
            last_start = s["start"]

    return masked


def sanitize_text(
    text: str,
    strategy: str = "hybrid",
    model_dir: Path | str | None = None,
) -> str:
    """Convenience pipeline that detects entities and masks them in one call.

    Parameters
    ----------
    text:
        Raw text to sanitize.
    strategy:
        One of ``'hybrid'`` (default) or ``'regex'``.
    model_dir:
        Optional path to fine-tuned model checkpoint.

    Returns
    -------
    str
        Sanitized text with redaction tokens.
    """
    if strategy == "regex":
        spans = extract_regex_entities(text)
    else:
        spans = extract_hybrid_entities(text, model_dir=model_dir)

    return mask_entities(text, spans)
