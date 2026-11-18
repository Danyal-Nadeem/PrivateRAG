"""Unit tests for Phase 5 End-to-End Evaluation."""
from __future__ import annotations

import pytest
from eval.e2e_eval import (
    load_labeled_sensitive_values,
    find_leaked_entities,
)


class TestE2EEvalComponents:

    def test_load_labeled_manifest(self):
        manifest = load_labeled_sensitive_values()
        assert len(manifest) > 0
        # Check an executive or HR doc structure
        exec_docs = [k for k in manifest if "exec" in k]
        assert len(exec_docs) > 0
        doc = manifest[exec_docs[0]]
        assert "access_level" in doc
        assert "entities" in doc
        assert isinstance(doc["entities"], list)

    def test_find_leaked_entities_detects_exact_match(self):
        entities_pool = [
            ("john.doe@company.com", "EMAIL"),
            ("0321-1234567", "PHONE"),
            ("EMP-99881", "EMP_ID"),
        ]
        text = "Please reach out to john.doe@company.com or call 0321-1234567."
        leaks = find_leaked_entities(text, entities_pool)
        assert len(leaks) == 2
        labels = {item["label"] for item in leaks}
        assert "EMAIL" in labels
        assert "PHONE" in labels
        assert "EMP_ID" not in labels

    def test_find_leaked_entities_empty_when_masked(self):
        entities_pool = [
            ("john.doe@company.com", "EMAIL"),
            ("0321-1234567", "PHONE"),
        ]
        masked_text = "Please reach out to [EMAIL_REDACTED] or call [PHONE_REDACTED]."
        leaks = find_leaked_entities(masked_text, entities_pool)
        assert len(leaks) == 0

    def test_find_leaked_entities_ignores_short_tokens(self):
        entities_pool = [
            ("PK", "COUNTRY"),
            ("a", "SHORT"),
        ]
        text = "PK is a country"
        leaks = find_leaked_entities(text, entities_pool)
        assert len(leaks) == 0
