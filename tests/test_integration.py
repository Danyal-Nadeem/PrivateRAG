"""Integration tests for Phase 4: Output sanitization in /ask and API integration.

Tests the ask() function directly (no live HTTP server needed) plus CORS headers.

Verifies:
1. /ask endpoint applies sanitize_text before returning — raw sensitive values never leave.
2. CORS middleware allows cross-origin requests (checked via TestClient options call).
3. Full credential login → ask flow with sanitized output.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

# Must be set BEFORE any src imports so TESTING=1 bypasses embedder warm-up
os.environ["TESTING"] = "1"
os.environ["SECRET_KEY"] = "integration-test-secret-key-at-least-32-chars-long"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.api.main import ask, AskRequest, app
from src.api.auth import create_token, get_current_role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hits(text: str = "some chunk content"):
    return [
        {
            "doc_id": "DOC-INT-001",
            "access_level": "internal",
            "chunk_index": 0,
            "title": "Employee Directory",
            "text": text,
            "score": 0.88,
        }
    ]


def _mock_llm(answer: str):
    m = MagicMock()
    m.generate.return_value = answer
    return m


# ---------------------------------------------------------------------------
# 1. Output sanitization: sensitive values never leave /ask
# ---------------------------------------------------------------------------

class TestAskOutputSanitization:
    """Direct calls to ask() verify that sanitize_text is wired correctly."""

    def _call_ask(self, raw_answer: str, role: str = "employee"):
        token = create_token(role)
        actual_role = get_current_role(f"Bearer {token}")
        with patch("src.api.main.search", return_value=_make_hits()), \
             patch("src.api.main.get_llm_provider", return_value=_mock_llm(raw_answer)):
            return ask(AskRequest(query="test query"), role=actual_role)

    def test_email_is_redacted(self):
        resp = self._call_ask("Contact alice@corp.com for details.")
        assert "[EMAIL_REDACTED]" in resp.answer
        assert "alice@corp.com" not in resp.answer

    def test_phone_us_is_redacted(self):
        resp = self._call_ask("Dial +1-703-621-9117 for support.")
        assert "[PHONE_REDACTED]" in resp.answer
        assert "703-621-9117" not in resp.answer

    def test_phone_pakistani_is_redacted(self):
        resp = self._call_ask("Call 0322-5107245 to reach HR.")
        assert "[PHONE_REDACTED]" in resp.answer
        assert "0322-5107245" not in resp.answer

    def test_emp_id_is_redacted(self):
        resp = self._call_ask("Employee record: EMP-99887.", role="hr")
        assert "[EMP_ID_REDACTED]" in resp.answer
        assert "EMP-99887" not in resp.answer

    def test_salary_usd_is_redacted(self):
        resp = self._call_ask("Compensation is $125,000 per year.", role="hr")
        assert "[SALARY_REDACTED]" in resp.answer
        assert "$125,000" not in resp.answer

    def test_salary_pkr_is_redacted(self):
        resp = self._call_ask("Manager base pay is PKR 450,000 per month.", role="hr")
        assert "[SALARY_REDACTED]" in resp.answer
        assert "PKR 450,000" not in resp.answer

    def test_multiple_entities_all_redacted(self):
        raw = "Contact alice@corp.com or EMP-12345. Salary: $95,000. Phone: 0333-1234567."
        resp = self._call_ask(raw, role="hr")
        assert "[EMAIL_REDACTED]" in resp.answer
        assert "[EMP_ID_REDACTED]" in resp.answer
        assert "[SALARY_REDACTED]" in resp.answer
        assert "[PHONE_REDACTED]" in resp.answer
        # Raw values must be gone
        assert "alice@corp.com" not in resp.answer
        assert "EMP-12345" not in resp.answer
        assert "$95,000" not in resp.answer
        assert "0333-1234567" not in resp.answer

    def test_non_sensitive_text_untouched(self):
        raw = "The office is closed on New Year's Day and Independence Day."
        resp = self._call_ask(raw, role="intern")
        assert resp.answer == raw
        assert "REDACTED" not in resp.answer

    def test_retrieved_docs_metadata_preserved(self):
        resp = self._call_ask("Some answer.", role="employee")
        assert len(resp.retrieved_documents) == 1
        assert resp.retrieved_documents[0].doc_id == "DOC-INT-001"
        assert resp.retrieved_documents[0].access_level == "internal"
        assert resp.role == "employee"

    def test_role_comes_from_jwt_not_body(self):
        """RBAC: role in answer metadata must equal the JWT role, never body."""
        resp = self._call_ask("Some info.", role="admin")
        assert resp.role == "admin"


# ---------------------------------------------------------------------------
# 2. CORS middleware: allow React dev server cross-origin requests
# ---------------------------------------------------------------------------

class TestCORSMiddleware:
    """Verify CORS middleware is registered on the FastAPI app."""

    def test_cors_middleware_is_registered(self):
        """CORS middleware must be in the app's middleware stack."""
        from starlette.middleware.cors import CORSMiddleware
        middleware_types = [type(m) for m in app.user_middleware]
        # FastAPI wraps middleware — check by class name in the middleware stack
        stack_repr = str(app.middleware_stack)
        assert "CORSMiddleware" in stack_repr or any(
            "cors" in str(m).lower() for m in app.user_middleware
        )

    def test_cors_allow_origin_configured(self):
        """Origins must be explicit (no wildcard) and include the Vite dev server."""
        for m in app.user_middleware:
            if "CORSMiddleware" in str(m):
                origins = getattr(m, "kwargs", {}).get("allow_origins", [])
                assert "*" not in origins, "Wildcard CORS origin is not allowed"
                assert "http://localhost:5173" in origins
                return
        pytest.fail("CORSMiddleware not found in app middleware stack")

    def test_cors_preflight_allows_frontend_origin(self):
        """Preflight from the Vite dev origin must be accepted."""
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            resp = client.options(
                "/ask",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization,Content-Type",
                },
            )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_preflight_rejects_unknown_origin(self):
        """Preflight from an arbitrary origin must NOT be granted access."""
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            resp = client.options(
                "/ask",
                headers={
                    "Origin": "http://evil.example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ---------------------------------------------------------------------------
# 3. Full login → ask flow (credential-based, server-assigned role)
# ---------------------------------------------------------------------------

class TestEndToEndLoginAsk:

    def test_login_then_sanitized_ask(self):
        """Credential login issues a valid JWT → ask uses it → answer redacted."""
        from src.api.main import login, LoginRequest
        login_resp = login(LoginRequest(username="hr", password="HrSpec#2026"))
        assert login_resp.role == "hr"
        actual_role = get_current_role(f"Bearer {login_resp.token}")
        raw_ans = "Manager pay PKR 450,000. Email: hr.ops@company.com. ID: EMP-77001."
        with patch("src.api.main.search", return_value=_make_hits()), \
             patch("src.api.main.get_llm_provider", return_value=_mock_llm(raw_ans)):
            resp = ask(AskRequest(query="manager details"), role=actual_role)
        assert resp.role == "hr"
        assert "[SALARY_REDACTED]" in resp.answer
        assert "[EMAIL_REDACTED]" in resp.answer
        assert "[EMP_ID_REDACTED]" in resp.answer
        assert "PKR 450,000" not in resp.answer
        assert "hr.ops@company.com" not in resp.answer
        assert "EMP-77001" not in resp.answer

    def test_login_wrong_password_returns_401(self):
        from fastapi import HTTPException
        from src.api.main import login, LoginRequest
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(username="admin", password="wrong-password"))
        assert exc_info.value.status_code == 401

    def test_login_cannot_self_assign_role(self):
        """An intern account must get intern clearance even if it asks for admin."""
        from src.api.main import login, LoginRequest
        resp = login(LoginRequest.model_validate(
            {"username": "intern", "password": "Intern#2026", "role": "admin"}
        ))
        assert resp.role == "intern"
        assert get_current_role(f"Bearer {resp.token}") == "intern"

    def test_passwordless_login_role_endpoint_removed(self):
        """The old privilege-escalation endpoint must no longer exist."""
        routes = {r.path for r in app.routes}
        assert "/login-role" not in routes
        assert "/login" in routes

    def test_strict_dlp_unavailable_returns_503(self, monkeypatch):
        """NER_STRICT=1 with no checkpoint → deliberate 503, never a raw answer."""
        from fastapi import HTTPException
        import src.ner.infer as infer
        from src.api.main import login, LoginRequest

        monkeypatch.setenv("NER_STRICT", "1")
        # Force a fresh model-load attempt against the (absent) checkpoint.
        monkeypatch.setattr(infer, "_model", None)
        monkeypatch.setattr(infer, "_tokenizer", None)
        monkeypatch.setattr(infer, "_model_load_attempted", False)

        login_resp = login(LoginRequest(username="hr", password="HrSpec#2026"))
        actual_role = get_current_role(f"Bearer {login_resp.token}")
        raw_ans = "Manager pay PKR 450,000. Email: hr.ops@company.com."
        with patch("src.api.main.search", return_value=_make_hits()), \
             patch("src.api.main.get_llm_provider", return_value=_mock_llm(raw_ans)):
            with pytest.raises(HTTPException) as exc_info:
                ask(AskRequest(query="manager details"), role=actual_role)
        assert exc_info.value.status_code == 503
        assert "fail-closed" in exc_info.value.detail
        # The unsanitized answer must never escape in the error payload.
        assert "PKR 450,000" not in str(exc_info.value.detail)

    def test_non_strict_dlp_unavailable_still_answers(self, monkeypatch):
        """Default (non-strict) mode degrades to regex-only, still HTTP 200."""
        import src.ner.infer as infer
        from src.api.main import login, LoginRequest

        monkeypatch.delenv("NER_STRICT", raising=False)
        monkeypatch.setattr(infer, "_model", None)
        monkeypatch.setattr(infer, "_tokenizer", None)
        monkeypatch.setattr(infer, "_model_load_attempted", False)

        login_resp = login(LoginRequest(username="hr", password="HrSpec#2026"))
        actual_role = get_current_role(f"Bearer {login_resp.token}")
        raw_ans = "Manager pay PKR 450,000."
        with patch("src.api.main.search", return_value=_make_hits()), \
             patch("src.api.main.get_llm_provider", return_value=_mock_llm(raw_ans)):
            resp = ask(AskRequest(query="manager details"), role=actual_role)
        assert "[SALARY_REDACTED]" in resp.answer
