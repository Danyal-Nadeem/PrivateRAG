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
        infer_mod.reset_model_cache()
        monkeypatch.delenv("NER_STRICT", raising=False)
        yield
        infer_mod.reset_model_cache()

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


# ---------------------------------------------------------------------------
# 7. Checkpoint hot-reload (a failed load retries when the file appears)
# ---------------------------------------------------------------------------

class TestModelHotReload:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        import src.ner.infer as infer_mod
        infer_mod.reset_model_cache()
        yield
        infer_mod.reset_model_cache()

    def test_failed_missing_checkpoint_stays_cached_without_change(self, tmp_path):
        """No checkpoint present -> the failure is cached, no repeated probing."""
        import src.ner.infer as infer_mod
        model_dir = tmp_path / "distilbert-ner"
        assert infer_mod._load_model(model_dir) == (None, None)
        assert infer_mod._model_load_attempted is True
        assert infer_mod._load_model(model_dir) == (None, None)

    def test_checkpoint_appearing_after_failure_is_retried(self, tmp_path, monkeypatch):
        """Dropping config.json in after a failed load triggers a reload."""
        import src.ner.infer as infer_mod
        model_dir = tmp_path / "distilbert-ner"
        assert infer_mod._load_model(model_dir) == (None, None)

        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        class FakeModel:
            def eval(self):
                return self

        loaded = {"tokenizer": 0, "model": 0}

        def fake_tok(path):
            loaded["tokenizer"] += 1
            return "tok"

        def fake_model(path):
            loaded["model"] += 1
            return FakeModel()

        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tok)
        monkeypatch.setattr(
            "transformers.AutoModelForTokenClassification.from_pretrained", fake_model
        )

        model, tokenizer = infer_mod._load_model(model_dir)
        assert model is not None
        assert tokenizer == "tok"
        assert loaded == {"tokenizer": 1, "model": 1}

        # Loaded successfully -> subsequent calls reuse the cached instance.
        model2, tokenizer2 = infer_mod._load_model(model_dir)
        assert model2 is model and tokenizer2 is tokenizer
        assert loaded == {"tokenizer": 1, "model": 1}


# ---------------------------------------------------------------------------
# 8. Pretrained baseline retry cooldown (transient failures are retryable)
# ---------------------------------------------------------------------------

class TestPretrainedBaselineRetry:
    @pytest.fixture(autouse=True)
    def _cleanup(self, monkeypatch):
        import src.ner.pretrained_baseline as pb
        pb.reset_baseline_cache()
        # Take the module out of offline mode for these tests.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        monkeypatch.delenv("TESTING", raising=False)
        yield
        pb.reset_baseline_cache()

    @staticmethod
    def _install_fake_transformers(monkeypatch, pipeline_factory):
        """Swap the transformers package for a stub in sys.modules.

        transformers replaces itself with a lazy module wrapper at import
        time, so patching a test-local `transformers` binding can hit a
        stale object while `from transformers import pipeline` reads the
        live sys.modules entry. The stub sidesteps the lazy loader entirely.
        """
        import sys
        import types

        fake = types.ModuleType("transformers")
        fake.pipeline = pipeline_factory
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return fake

    def test_transient_failure_does_not_retry_within_cooldown(self, monkeypatch):
        """A network failure must not cause a download attempt per record."""
        import src.ner.pretrained_baseline as pb
        attempts = {"n": 0}

        def boom(*a, **k):
            attempts["n"] += 1
            raise RuntimeError("network down")

        self._install_fake_transformers(monkeypatch, boom)
        assert pb.extract_pretrained_entities("Alice works at Acme") == []
        assert attempts["n"] == 1
        # Inside the cooldown window: no second attempt.
        assert pb.extract_pretrained_entities("Alice works at Acme") == []
        assert attempts["n"] == 1

    def test_transient_failure_retries_after_cooldown(self, monkeypatch):
        """Once the cooldown passes, the pipeline is attempted again."""
        import src.ner.pretrained_baseline as pb
        attempts = {"n": 0}

        def boom(*a, **k):
            attempts["n"] += 1
            raise RuntimeError("network down")

        fake = self._install_fake_transformers(monkeypatch, boom)
        assert pb.extract_pretrained_entities("Alice works at Acme") == []
        assert attempts["n"] == 1

        # Jump the clock past the cooldown and provide a working pipeline.
        base = pb._now()
        monkeypatch.setattr(pb, "_now", lambda: base + 61.0)

        class _FakePipeline:
            """pipeline() is a factory: it must return a callable object."""

            def __call__(self, text):
                return [{"entity_group": "PER", "start": 0, "end": 5}]

        fake.pipeline = lambda *a, **k: _FakePipeline()
        spans = pb.extract_pretrained_entities("Alice works at Acme")
        assert [s["label"] for s in spans] == ["PERSON"]
        assert attempts["n"] == 1  # boom not called again; fake pipeline used

    def test_offline_mode_disables_permanently(self, monkeypatch):
        """HF_HUB_OFFLINE remains a hard kill-switch: no retries at all."""
        import src.ner.pretrained_baseline as pb
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        attempts = {"n": 0}

        def boom(*a, **k):
            attempts["n"] += 1
            raise RuntimeError("should not be called")

        self._install_fake_transformers(monkeypatch, boom)
        assert pb.extract_pretrained_entities("Alice works at Acme") == []
        assert attempts["n"] == 0
        assert pb._pipeline_failed is True
