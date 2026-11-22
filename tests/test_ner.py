"""Pytest test suite for Phase 3 Named Entity Recognition and Sanitization.

Tests:
1. Regex extraction for pattern-based entities (EMAIL, PHONE, EMP_ID, SALARY)
2. Negative confusers (ticket IDs, invoice numbers, non-phone digits)
3. Non-destructive right-to-left masking engine
4. Hybrid filter arbitration logic (Regex wins on pattern, Model on context)
5. Model inference fallback when weights are not yet present locally

Run with:
    python -m pytest tests/test_ner.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.regex_filter import (
    extract_regex_entities,
    EMAIL_PATTERN,
    PHONE_PATTERN,
    EMP_ID_PATTERN,
    SALARY_PATTERN,
)
from src.ner.masking import mask_entities, sanitize_text
from src.ner.hybrid_filter import extract_hybrid_entities
from src.ner.infer import extract_model_entities


# ---------------------------------------------------------------------------
# 1. Regex Entity Extraction
# ---------------------------------------------------------------------------

class TestRegexFilter:
    def test_extract_email(self):
        text = "Send reports to admin.support@enterprise.org before 5 PM."
        spans = extract_regex_entities(text)
        assert len(spans) == 1
        assert spans[0]["label"] == "EMAIL"
        assert spans[0]["text"] == "admin.support@enterprise.org"
        assert text[spans[0]["start"]:spans[0]["end"]] == "admin.support@enterprise.org"

    def test_extract_emp_id_formats(self):
        # Format 1: EMP-12345
        t1 = "Employee EMP-89012 submitted the quarterly review."
        s1 = extract_regex_entities(t1)
        assert any(s["label"] == "EMP_ID" and s["text"] == "EMP-89012" for s in s1)

        # Format 2: EMP12345
        t2 = "Badge EMP91763 logged into the main server."
        s2 = extract_regex_entities(t2)
        assert any(s["label"] == "EMP_ID" and s["text"] == "EMP91763" for s in s2)

        # Format 3: E-12345
        t3 = "Consultant E-98651 has access to floor 3."
        s3 = extract_regex_entities(t3)
        assert any(s["label"] == "EMP_ID" and s["text"] == "E-98651" for s in s3)

    def test_emp_id_ignores_hard_negatives(self):
        # Ticket IDs, Invoices, Orders, SKUs must NOT be extracted as EMP_ID
        text = "Ticket TKT-48213 and invoice INV-90214 reference purchase order ORD-55102."
        spans = extract_regex_entities(text)
        emp_spans = [s for s in spans if s["label"] == "EMP_ID"]
        assert len(emp_spans) == 0, f"False positive EMP_ID in negatives: {emp_spans}"

    def test_extract_phone_formats(self):
        # US formats
        t_us = "Call (217) 289-7912 or direct line 703-621-9117."
        s_us = extract_regex_entities(t_us)
        phones_us = [s["text"] for s in s_us if s["label"] == "PHONE"]
        assert "(217) 289-7912" in phones_us
        assert "703-621-9117" in phones_us

        # Pakistani formats
        t_pk = "Contact mobile 0322-5107245 or +92 320 6292423 in Lahore."
        s_pk = extract_regex_entities(t_pk)
        phones_pk = [s["text"] for s in s_pk if s["label"] == "PHONE"]
        assert "0322-5107245" in phones_pk
        assert "+92 320 6292423" in phones_pk

    def test_extract_salary_currencies(self):
        t1 = "Base salary is $125,000 with a bonus."
        assert any(s["label"] == "SALARY" and s["text"] == "$125,000" for s in extract_regex_entities(t1))

        t2 = "Approved amount: USD 95,000 for foreign placement."
        assert any(s["label"] == "SALARY" and s["text"] == "USD 95,000" for s in extract_regex_entities(t2))

        t3 = "Compensation: PKR 350,000 per month."
        assert any(s["label"] == "SALARY" and s["text"] == "PKR 350,000" for s in extract_regex_entities(t3))

        t4 = "Executive package ratified at Rs. 1,200,000 annually."
        assert any(s["label"] == "SALARY" and s["text"] == "Rs. 1,200,000" for s in extract_regex_entities(t4))

        t5 = "Offer letter states $95k baseline."
        assert any(s["label"] == "SALARY" and s["text"] == "$95k" for s in extract_regex_entities(t5))

    def test_phone_not_matched_inside_longer_digit_run(self):
        """US branch must not start mid-number (e.g. invoice refs)."""
        text = "Invoice ref 1234-567-8901 was reconciled."
        phones = [s["text"] for s in extract_regex_entities(text) if s["label"] == "PHONE"]
        assert "234-567-8901" not in phones
        assert phones == []

    def test_phone_still_matches_standalone_number(self):
        """The boundary guard must not break genuine numbers."""
        text = "Direct line 234-567-8901 after hours."
        phones = [s["text"] for s in extract_regex_entities(text) if s["label"] == "PHONE"]
        assert "234-567-8901" in phones

    def test_salary_rs_without_dot(self):
        t = "Stipend: Rs 45,000 per month."
        assert any(s["label"] == "SALARY" and s["text"] == "Rs 45,000" for s in extract_regex_entities(t))

    def test_salary_decimal_amount_fully_covered(self):
        """$180,000.00 must be masked in full — no '.00' remnant."""
        t = "Offer: $180,000.00 annual base."
        spans = [s for s in extract_regex_entities(t) if s["label"] == "SALARY"]
        assert any(s["text"] == "$180,000.00" for s in spans)


# ---------------------------------------------------------------------------
# 2. Token Masking Engine
# ---------------------------------------------------------------------------

class TestMaskingEngine:
    def test_single_entity_masked(self):
        text = "Reach out to user@example.com immediately."
        spans = [{"start": 13, "end": 29, "label": "EMAIL", "text": "user@example.com"}]
        masked = mask_entities(text, spans)
        assert masked == "Reach out to [EMAIL_REDACTED] immediately."

    def test_multiple_entities_right_to_left_preserves_offsets(self):
        text = "Staff EMP-12345 earns $85,000 and uses user@corp.org for comms."
        spans = extract_regex_entities(text)
        assert len(spans) == 3
        masked = mask_entities(text, spans)
        assert "[EMP_ID_REDACTED]" in masked
        assert "[SALARY_REDACTED]" in masked
        assert "[EMAIL_REDACTED]" in masked
        # Crucial: verify no offset drift corrupted surrounding text
        assert masked.startswith("Staff [EMP_ID_REDACTED] earns [SALARY_REDACTED]")
        assert masked.endswith("for comms.")

    def test_sanitize_text_convenience(self):
        text = "Notify EMP-88909 at akhuhro@corp.acme.com via phone (518) 670-9785."
        masked = sanitize_text(text, strategy="regex")
        assert "EMP-88909" not in masked
        assert "akhuhro@corp.acme.com" not in masked
        assert "(518) 670-9785" not in masked
        assert "[EMP_ID_REDACTED]" in masked
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked

    def test_empty_text_returns_empty(self):
        assert mask_entities("", []) == ""
        assert sanitize_text("", strategy="regex") == ""


# ---------------------------------------------------------------------------
# 3. Hybrid Filter Decision Arbitration
# ---------------------------------------------------------------------------

class TestHybridFilter:
    def test_regex_wins_for_pattern_entities(self):
        """When regex finds EMAIL and model proposes PERSON over the same span, regex wins."""
        text = "Email contact: dylan.80104@enterprise.org"
        with patch("src.ner.hybrid_filter.extract_model_entities") as mock_model:
            # Simulate a flawed model predicting PERSON over the email span
            mock_model.return_value = [
                {"start": 15, "end": 41, "label": "PERSON", "text": "dylan.80104@enterprise.org"}
            ]
            spans = extract_hybrid_entities(text, use_model=True)
            labels = [s["label"] for s in spans]
            assert "EMAIL" in labels
            assert "PERSON" not in labels

    def test_model_wins_for_person_and_address(self):
        """Model entities for PERSON and ADDRESS are retained when non-overlapping with regex."""
        text = "Meeting with Dr. Azhar Memon in Lahore."
        with patch("src.ner.hybrid_filter.extract_model_entities") as mock_model:
            mock_model.return_value = [
                {"start": 13, "end": 28, "label": "PERSON", "text": "Dr. Azhar Memon"}
            ]
            spans = extract_hybrid_entities(text, use_model=True)
            assert len(spans) == 1
            assert spans[0]["label"] == "PERSON"
            assert spans[0]["text"] == "Dr. Azhar Memon"


# ---------------------------------------------------------------------------
# 4. Model Inference Fallback (Runs safely without local weights)
# ---------------------------------------------------------------------------

class TestModelInferenceSafety:
    def test_infer_empty_when_no_checkpoint(self):
        # Point to a nonexistent directory
        spans = extract_model_entities("Hello world", model_dir="nonexistent/dir")
        assert spans == []


# ---------------------------------------------------------------------------
# 6. DLP degradation must be loud and observable (never silent)
# ---------------------------------------------------------------------------

class TestDlpDegradationVisibility:
    @pytest.fixture(autouse=True)
    def _reset_model_cache(self, monkeypatch):
        """Reset infer module cache so each test exercises the load path."""
        import src.ner.infer as infer_mod
        monkeypatch.setattr(infer_mod, "_model", None)
        monkeypatch.setattr(infer_mod, "_tokenizer", None)
        monkeypatch.setattr(infer_mod, "_model_load_attempted", False)
        monkeypatch.delenv("NER_STRICT", raising=False)
        yield

    def test_missing_checkpoint_logs_warning(self, caplog):
        import logging
        import src.ner.infer as infer_mod
        with caplog.at_level(logging.WARNING, logger="privaterag.ner"):
            infer_mod.extract_model_entities(
                "John Smith lives at 12 Main Street.",
                model_dir="nonexistent/dir",
            )
        assert any("DLP DEGRADED" in r.message for r in caplog.records)

    def test_model_available_false_without_checkpoint(self):
        from src.ner.infer import model_available
        assert model_available(model_dir="nonexistent/dir") is False

    def test_strict_mode_fails_closed(self, monkeypatch):
        monkeypatch.setenv("NER_STRICT", "1")
        from src.ner.infer import NerModelUnavailableError
        with pytest.raises(NerModelUnavailableError):
            extract_model_entities("John Smith", model_dir="nonexistent/dir")

    def test_non_strict_mode_degrades_to_empty(self):
        spans = extract_model_entities("John Smith", model_dir="nonexistent/dir")
        assert spans == []
