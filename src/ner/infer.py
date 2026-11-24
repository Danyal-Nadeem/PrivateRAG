"""Model inference engine returning exact CHARACTER-offset entity spans.

Loads fine-tuned DistilBERT from models/distilbert-ner/ if present.
Maps token-level predictions back to raw character spans using offset mapping.

Security note: when the checkpoint is missing the DLP layer degrades to
regex-only (PERSON/ADDRESS are NOT masked). This degradation is logged
loudly and exposed via model_available() / /health. Set NER_STRICT=1 to
fail closed (raise) instead of degrading.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("privaterag.ner")

# Map label IDs to our entity labels
ID2LABEL: dict[int, str] = {
    0: "O",
    1: "B-PERSON",
    2: "I-PERSON",
    3: "B-EMAIL",
    4: "I-EMAIL",
    5: "B-PHONE",
    6: "I-PHONE",
    7: "B-EMP_ID",
    8: "I-EMP_ID",
    9: "B-ADDRESS",
    10: "I-ADDRESS",
    11: "B-SALARY",
    12: "I-SALARY",
}

LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-ner"

_model = None
_tokenizer = None
_model_load_attempted = False
_checkpoint_mtime_at_attempt: float | None = None


class NerModelUnavailableError(RuntimeError):
    """Raised in strict mode (NER_STRICT=1) when the DLP model cannot run."""


def reset_model_cache() -> None:
    """Forget the cached load result so the checkpoint is re-attempted.

    Useful for tests and for hot-reloading a checkpoint that is dropped
    into models/distilbert-ner/ while the server is running.
    """
    global _model, _tokenizer, _model_load_attempted, _checkpoint_mtime_at_attempt
    _model = None
    _tokenizer = None
    _model_load_attempted = False
    _checkpoint_mtime_at_attempt = None


def _checkpoint_changed(target_dir: Path) -> bool:
    """Return True if config.json appeared/changed since the last attempt."""
    config = target_dir / "config.json"
    if not config.exists():
        return False
    mtime = config.stat().st_mtime
    return _checkpoint_mtime_at_attempt is None or mtime != _checkpoint_mtime_at_attempt


def _load_model(model_dir: Path | str | None = None):
    global _model, _tokenizer, _model_load_attempted, _checkpoint_mtime_at_attempt
    target_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    # Once loaded successfully, keep the cached instance. If the previous
    # attempt failed, allow a retry when the checkpoint file itself changed
    # (e.g. it was copied in after server start) — otherwise stay failed.
    if _model_load_attempted:
        if _model is not None or not _checkpoint_changed(target_dir):
            return _model, _tokenizer
    _model_load_attempted = True

    if not (target_dir / "config.json").exists():
        # Model not trained yet — degrade loudly, never silently.
        logger.warning(
            "DLP DEGRADED: fine-tuned NER checkpoint not found at %s. "
            "PERSON and ADDRESS entities will NOT be masked (regex-only mode). "
            "Train via notebooks/train_colab.ipynb and place the checkpoint "
            "in models/distilbert-ner/, or set NER_STRICT=1 to fail closed.",
            target_dir,
        )
        return None, None

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
        _model = AutoModelForTokenClassification.from_pretrained(str(target_dir))
        _model.eval()
    except Exception as exc:
        _model = None
        _tokenizer = None
        logger.error(
            "DLP DEGRADED: failed to load NER checkpoint from %s (%s). "
            "PERSON and ADDRESS entities will NOT be masked.",
            target_dir, exc,
        )

    if (target_dir / "config.json").exists():
        _checkpoint_mtime_at_attempt = (target_dir / "config.json").stat().st_mtime

    return _model, _tokenizer


def model_available(model_dir: Path | str | None = None) -> bool:
    """Return True if the fine-tuned DLP model is loaded and usable."""
    model, tokenizer = _load_model(model_dir)
    return model is not None and tokenizer is not None


def extract_model_entities(
    text: str,
    model_dir: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Run model inference on *text* and return character-offset spans.

    Parameters
    ----------
    text:
        Raw input string.
    model_dir:
        Optional custom path to model checkpoint.

    Returns
    -------
    List[Dict[str, Any]]
        Each element is a dict:
        ``{"start": int, "end": int, "label": str, "text": str}``
    """
    if not text.strip():
        return []

    model, tokenizer = _load_model(model_dir)
    if model is None or tokenizer is None:
        if os.getenv("NER_STRICT") == "1":
            raise NerModelUnavailableError(
                "NER_STRICT=1: fine-tuned DLP model is unavailable. "
                "Refusing to process text without contextual entity masking."
            )
        return []

    import torch

    inputs = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1).tolist()

    spans: List[Dict[str, Any]] = []
    current_label = None
    current_start = None
    current_end = None

    for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens (start == end == 0)
        if start == end:
            continue

        tag = ID2LABEL.get(pred_id, "O")

        if tag.startswith("B-"):
            # Flush previous span
            if current_label and current_start is not None and current_end is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": current_label,
                    "text": text[current_start:current_end].strip(),
                })
            current_label = tag[2:]
            current_start = start
            current_end = end

        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label == label and current_start is not None:
                # Extend current span
                current_end = end
            else:
                # Treat as new span if B- was missed
                if current_label and current_start is not None and current_end is not None:
                    spans.append({
                        "start": current_start,
                        "end": current_end,
                        "label": current_label,
                        "text": text[current_start:current_end].strip(),
                    })
                current_label = label
                current_start = start
                current_end = end
        else:
            # tag == "O"
            if current_label and current_start is not None and current_end is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": current_label,
                    "text": text[current_start:current_end].strip(),
                })
                current_label = None
                current_start = None
                current_end = None

    if current_label and current_start is not None and current_end is not None:
        spans.append({
            "start": current_start,
            "end": current_end,
            "label": current_label,
            "text": text[current_start:current_end].strip(),
        })

    # Sort and clean
    spans = [s for s in spans if s["text"]]
    spans.sort(key=lambda s: s["start"])
    return spans
