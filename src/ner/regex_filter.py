"""Regex baseline filter for pattern-based entities.

Handles:
- EMAIL: standard RFC email pattern
- PHONE: US formats and Pakistani (+92 / 03xx) formats
- EMP_ID: EMP-12345, EMP12345, E-12345
- SALARY: Currency + amount ($125,000, USD 95,000, PKR 350,000, Rs. 1,200,000,
  Rs 45,000, $180,000.00, $95k)

Returns character-offset spans:
    [{"start": int, "end": int, "label": str, "text": str}]
"""
from __future__ import annotations

import re
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Compiled Regex Patterns
# ---------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

EMP_ID_PATTERN = re.compile(
    r"\b(?:EMP-\d{5}|EMP\d{5}|E-\d{5})\b"
)

# Pakistani phone patterns:
# +92-42-376647119, +92 320 6292423, 0322-5107245, 0320 8339683, etc.
# US phone patterns:
# (217) 289-7912, 703-621-9117, +1-518-271-1750, 319.467.2415, 555-123-4567
# The US branch carries a (?<![\d(]) guard so it cannot start matching in the
# middle of a longer digit run (e.g. invoice refs like 1234-567-8901).
PHONE_PATTERN = re.compile(
    r"(?:"
    # Pakistani international with +92
    r"\+92[- ](?:42[- ]\d{7,9}|3\d{2}[- ]\d{7})"
    # Pakistani local mobile starting with 03xx
    r"|\b03\d{2}[- ]?\d{7}\b"
    # US standard formats: +1-xxx-xxx-xxxx, (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx
    r"|(?<![\d(])(?:\+1[-. ])?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"
    r")"
)

# Salary patterns: currency symbol or code followed by formatted number
# Currency amounts: $125,000, $95k, $180,000.00, USD 95,000, PKR 350,000,
# Rs. 1,200,000, Rs 45,000, GBP 85,000
SALARY_PATTERN = re.compile(
    r"(?:"
    r"\$(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?:k|K)?\b"
    r"|\b(?:USD|PKR|EUR|GBP)\s+(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\b"
    r"|\bRs\.?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\b"
    r")"
)


def extract_regex_entities(text: str) -> List[Dict[str, Any]]:
    """Scan *text* and return detected entity spans sorted by start offset.

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    List[Dict[str, Any]]
        Each element is a dict:
        ``{"start": int, "end": int, "label": str, "text": str}``
    """
    spans: List[Dict[str, Any]] = []

    # 1. EMAIL
    for m in EMAIL_PATTERN.finditer(text):
        spans.append({
            "start": m.start(),
            "end": m.end(),
            "label": "EMAIL",
            "text": m.group(),
        })

    # 2. EMP_ID
    for m in EMP_ID_PATTERN.finditer(text):
        spans.append({
            "start": m.start(),
            "end": m.end(),
            "label": "EMP_ID",
            "text": m.group(),
        })

    # 3. PHONE
    for m in PHONE_PATTERN.finditer(text):
        spans.append({
            "start": m.start(),
            "end": m.end(),
            "label": "PHONE",
            "text": m.group(),
        })

    # 4. SALARY
    for m in SALARY_PATTERN.finditer(text):
        spans.append({
            "start": m.start(),
            "end": m.end(),
            "label": "SALARY",
            "text": m.group(),
        })

    # Sort spans by start offset, and longer spans first if starts coincide
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

    # Non-overlapping resolution: keep earlier/longer span
    resolved: List[Dict[str, Any]] = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            resolved.append(s)
            last_end = s["end"]

    return resolved
