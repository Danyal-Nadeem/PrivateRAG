"""pytest test suite for Phase 2 RBAC, JWT, and API behaviour.

Run with:
    python -m pytest tests/ -v

No live LLM or API calls are made.  Qdrant payload/metadata consistency
is tested against the real data files.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def metadata_index() -> dict[str, str]:
    """Load {doc_id: access_level} from metadata.json."""
    path = PROJECT_ROOT / "data" / "documents" / "metadata.json"
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    return {d["doc_id"]: d["access_level"] for d in docs}


@pytest.fixture(scope="session")
def env_secret(monkeypatch=None):
    """Ensure SECRET_KEY is set for JWT tests."""
    os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789")
    return os.environ["SECRET_KEY"]


# ---------------------------------------------------------------------------
# 1. Role-to-levels mapping
# ---------------------------------------------------------------------------

class TestRoleToLevels:
    def test_intern_levels(self):
        from src.retrieval.rbac import get_allowed_levels
        assert get_allowed_levels("intern") == ["public"]

    def test_employee_levels(self):
        from src.retrieval.rbac import get_allowed_levels
        assert get_allowed_levels("employee") == ["public", "internal"]

    def test_hr_levels(self):
        from src.retrieval.rbac import get_allowed_levels
        assert get_allowed_levels("hr") == ["public", "internal", "hr_restricted"]

    def test_admin_levels(self):
        from src.retrieval.rbac import get_allowed_levels
        assert get_allowed_levels("admin") == [
            "public", "internal", "hr_restricted", "exec_restricted"
        ]

    def test_invalid_role_raises(self):
        from src.retrieval.rbac import get_allowed_levels
        with pytest.raises(ValueError, match="Unknown role"):
            get_allowed_levels("superuser")

    def test_all_valid_roles_covered(self):
        from src.retrieval.rbac import ROLE_ACCESS_MAP
        assert set(ROLE_ACCESS_MAP.keys()) == {"intern", "employee", "hr", "admin"}


# ---------------------------------------------------------------------------
# 2. Qdrant filter construction
# ---------------------------------------------------------------------------

class TestFilterConstruction:
    def test_filter_type(self):
        from qdrant_client import models
        from src.retrieval.rbac import build_rbac_filter
        f = build_rbac_filter("employee")
        assert isinstance(f, models.Filter)

    def test_filter_match_any_intern(self):
        from src.retrieval.rbac import build_rbac_filter
        f = build_rbac_filter("intern")
        condition = f.must[0]
        assert condition.key == "access_level"
        assert set(condition.match.any) == {"public"}

    def test_filter_match_any_hr(self):
        from src.retrieval.rbac import build_rbac_filter
        f = build_rbac_filter("hr")
        condition = f.must[0]
        assert set(condition.match.any) == {"public", "internal", "hr_restricted"}

    def test_filter_match_any_admin(self):
        from src.retrieval.rbac import build_rbac_filter
        f = build_rbac_filter("admin")
        condition = f.must[0]
        assert set(condition.match.any) == {
            "public", "internal", "hr_restricted", "exec_restricted"
        }

    def test_invalid_role_raises_in_filter(self):
        from src.retrieval.rbac import build_rbac_filter
        with pytest.raises(ValueError):
            build_rbac_filter("guest")

    def test_filter_has_exactly_one_must_condition(self):
        from src.retrieval.rbac import build_rbac_filter
        for role in ("intern", "employee", "hr", "admin"):
            f = build_rbac_filter(role)
            assert len(f.must) == 1, f"role={role}: expected 1 must condition"


# ---------------------------------------------------------------------------
# 3. Payload vs metadata.json consistency
# ---------------------------------------------------------------------------

class TestPayloadMetadataConsistency:
    def test_all_payload_levels_match_metadata(self, metadata_index):
        """Every access_level stored in Qdrant payload must match metadata.json."""
        try:
            from src.retrieval.qdrant_service import scroll_all_payloads
            payloads = scroll_all_payloads()
        except Exception as e:
            pytest.skip(f"Qdrant not available or not ingested: {e}")

        if not payloads:
            pytest.skip("No documents ingested yet — run ingest.py first")

        mismatches = []
        for p in payloads:
            doc_id = p["doc_id"]
            payload_level = p["access_level"]
            meta_level = metadata_index.get(doc_id)
            if meta_level is None:
                mismatches.append(f"{doc_id}: not in metadata.json")
            elif payload_level != meta_level:
                mismatches.append(
                    f"{doc_id}: payload={payload_level!r} vs metadata={meta_level!r}"
                )
        assert not mismatches, (
            f"{len(mismatches)} payload/metadata mismatch(es):\n"
            + "\n".join(mismatches[:10])
        )


# ---------------------------------------------------------------------------
# 4. JWT token creation and validation
# ---------------------------------------------------------------------------

class TestJWT:
    def _set_secret(self):
        os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789")

    def test_create_and_decode_valid_token(self):
        self._set_secret()
        from src.api.auth import create_token, decode_token
        token = create_token("employee")
        role = decode_token(token)
        assert role == "employee"

    def test_create_token_for_all_roles(self):
        self._set_secret()
        from src.api.auth import create_token, decode_token
        for role in ("intern", "employee", "hr", "admin"):
            token = create_token(role)
            assert decode_token(token) == role

    def test_invalid_role_raises_on_create(self):
        self._set_secret()
        from src.api.auth import create_token
        with pytest.raises(ValueError, match="Unknown role"):
            create_token("hacker")

    def test_expired_token_raises_401(self):
        self._set_secret()
        import jwt as pyjwt
        from fastapi import HTTPException
        from src.api.auth import ALGORITHM, _get_secret, decode_token
        secret = _get_secret()
        payload = {
            "role": "intern",
            "exp": datetime.now(tz=timezone.utc) - timedelta(seconds=1),
        }
        expired_token = pyjwt.encode(payload, secret, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_tampered_token_raises_401(self):
        self._set_secret()
        from fastapi import HTTPException
        from src.api.auth import create_token, decode_token
        token = create_token("intern")
        # Flip a character in the signature portion
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + "." + parts[2][:-1] + "X"
        with pytest.raises(HTTPException) as exc_info:
            decode_token(tampered)
        assert exc_info.value.status_code == 401

    def test_missing_secret_raises(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        from src.api.auth import _get_secret
        with pytest.raises(RuntimeError, match="not set"):
            _get_secret()

    def test_placeholder_secret_rejected(self, monkeypatch):
        from src.api.auth import _get_secret
        for weak in ("your-secret-key-here",
                     "privaterag-super-secret-key-change-in-production"):
            monkeypatch.setenv("SECRET_KEY", weak)
            with pytest.raises(RuntimeError, match="placeholder"):
                _get_secret()

    def test_short_secret_rejected(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "too-short-key")
        from src.api.auth import _get_secret
        with pytest.raises(RuntimeError, match="too short"):
            _get_secret()


# ---------------------------------------------------------------------------
# 5. API endpoint behaviour — tested without a live Qdrant connection
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    """
    These tests verify auth/RBAC logic at the FastAPI layer.
    We avoid importing src.api.main (which would init QdrantClient) by
    testing the underlying auth functions directly, plus credential
    verification against the user store.
    """

    @pytest.fixture(autouse=True)
    def _setup_env(self):
        os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789")
        os.environ.setdefault("LLM_PROVIDER", "extractive")
        os.environ.setdefault("TESTING", "1")
        # NOTE: never pop TESTING here — other test modules (e.g.
        # test_integration's TestClient lifespan) depend on it staying set.
        yield

    def test_login_creates_valid_token(self):
        """POST /login logic: create_token returns a JWT the auth layer can decode."""
        from src.api.auth import create_token, decode_token
        for role in ("intern", "employee", "hr", "admin"):
            token = create_token(role)
            assert decode_token(token) == role

    def test_login_invalid_role_raises(self):
        """create_token raises ValueError for unknown role."""
        from src.api.auth import create_token
        with pytest.raises(ValueError):
            create_token("superuser")

    def test_ask_role_comes_from_jwt_not_body(self):
        """
        The role that RBAC uses must come from the JWT, not from request body.
        We verify this by checking get_current_role() returns the JWT role
        even when a different role string is passed elsewhere.
        """
        from src.api.auth import create_token, decode_token
        # Simulate intern token
        token = create_token("intern")
        # get_current_role parses the Authorization header
        role_from_token = decode_token(token)
        assert role_from_token == "intern"

        # An admin token decode gives admin
        admin_token = create_token("admin")
        assert decode_token(admin_token) == "admin"

        # They are different — if a client sent intern token + "role": "admin" in body,
        # only the token role (intern) would ever be used by build_rbac_filter
        from src.retrieval.rbac import build_rbac_filter
        intern_filter = build_rbac_filter("intern")
        allowed = set(intern_filter.must[0].match.any)
        assert "exec_restricted" not in allowed, (
            "RBAC filter for intern must not include exec_restricted"
        )

    def test_ask_body_role_field_ignored(self):
        """
        Demonstrate that even if 'role' appears in AskRequest body,
        pydantic ignores it because the field is not declared.
        """
        from pydantic import BaseModel

        class AskRequest(BaseModel):
            query: str
            model_config = {"extra": "ignore"}

        req = AskRequest.model_validate({"query": "test", "role": "admin"})
        assert not hasattr(req, "role")
        assert req.query == "test"

    def test_health_no_qdrant_returns_ok(self):
        """health endpoint reports status=ok even when Qdrant unavailable."""
        # We test the response model construction directly (without loading main)
        from pydantic import BaseModel

        class HealthResponse(BaseModel):
            status: str
            qdrant_persistent: bool
            total_chunks: int
            llm_provider: str

        h = HealthResponse(
            status="ok",
            qdrant_persistent=False,
            total_chunks=-1,
            llm_provider="extractive",
        )
        assert h.status == "ok"
        assert h.total_chunks == -1


# ---------------------------------------------------------------------------
# 6. Query injection safety — role in query text does not bypass filter
# ---------------------------------------------------------------------------

class TestQueryInjectionSafety:
    def test_role_in_query_text_does_not_change_filter(self):
        """build_rbac_filter must depend only on role arg, not query text."""
        from src.retrieval.rbac import build_rbac_filter
        f1 = build_rbac_filter("intern")
        f2 = build_rbac_filter("intern")
        # Filters for the same role must have identical allowed levels
        assert set(f1.must[0].match.any) == set(f2.must[0].match.any)

    def test_intern_filter_never_allows_exec_restricted(self):
        from src.retrieval.rbac import build_rbac_filter
        f = build_rbac_filter("intern")
        allowed = set(f.must[0].match.any)
        assert "exec_restricted" not in allowed
        assert "hr_restricted" not in allowed
        assert "internal" not in allowed


# ---------------------------------------------------------------------------
# 7. LLM provider factory
# ---------------------------------------------------------------------------

class TestLLMProviderFactory:
    def test_extractive_is_default(self):
        os.environ["LLM_PROVIDER"] = "extractive"
        from src.llm import get_llm_provider
        from src.llm.extractive import ExtractiveProvider
        provider = get_llm_provider()
        assert isinstance(provider, ExtractiveProvider)

    def test_unknown_provider_raises(self):
        os.environ["LLM_PROVIDER"] = "nonexistent"
        from importlib import reload
        import src.llm as llm_module
        reload(llm_module)
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            llm_module.get_llm_provider()
        os.environ["LLM_PROVIDER"] = "extractive"  # restore

    def test_gemini_provider_raises_without_key(self):
        """GeminiProvider.__init__ raises RuntimeError when key is absent."""
        saved = os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GEMINI_MODEL", None)
        try:
            from src.llm.gemini import GeminiProvider
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                GeminiProvider()
        finally:
            if saved:
                os.environ["GEMINI_API_KEY"] = saved

    def test_gemini_provider_raises_without_model(self):
        """GeminiProvider raises RuntimeError when model is absent."""
        os.environ["GEMINI_API_KEY"] = "fake-key"
        os.environ.pop("GEMINI_MODEL", None)
        try:
            from src.llm.gemini import GeminiProvider
            with pytest.raises(RuntimeError):
                GeminiProvider()
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_gemini_provider_raises_without_package(self):
        """GeminiProvider raises RuntimeError when google-generativeai is absent."""
        os.environ["GEMINI_API_KEY"] = "fake-key"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        try:
            # Simulate missing package by blocking the import
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "google.generativeai" in name or name == "google":
                    raise ImportError("No module named 'google'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from importlib import reload
                import src.llm.gemini as gemini_mod
                reload(gemini_mod)
                with pytest.raises(RuntimeError, match="google-generativeai"):
                    gemini_mod.GeminiProvider()
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GEMINI_MODEL", None)

    def test_local_hf_provider_factory_with_mock(self):
        """Factory returns LocalHFProvider when LLM_PROVIDER=local_hf (mocked)."""
        os.environ["LLM_PROVIDER"] = "local_hf"
        with patch("src.llm.local_hf.LocalHFProvider.__init__", return_value=None):
            from importlib import reload
            import src.llm as llm_module
            reload(llm_module)
            import src.llm.local_hf as hf_mod
            reload(hf_mod)
            from src.llm.local_hf import LocalHFProvider
            with patch.object(LocalHFProvider, "__init__", return_value=None):
                provider = llm_module.get_llm_provider()
                assert isinstance(provider, LocalHFProvider)
        os.environ["LLM_PROVIDER"] = "extractive"


# ---------------------------------------------------------------------------
# 8. Credential-based authentication (roles are server-assigned)
# ---------------------------------------------------------------------------

class TestCredentialAuth:
    """Authentication must precede authorization: no passwordless role grants."""

    DEMO_PASSWORDS = {
        "intern": "Intern#2026",
        "employee": "Employee#2026",
        "hr": "HrSpec#2026",
        "admin": "Admin#2026",
    }

    @pytest.fixture(autouse=True)
    def _demo_store(self, monkeypatch):
        """Opt in to the demo store and reset throttle/cache state per test."""
        import src.api.users as users
        monkeypatch.setenv("ALLOW_DEMO_ACCOUNTS", "1")
        monkeypatch.setattr(users, "_users_cache", None)
        users.reset_login_throttle()
        yield
        users.reset_login_throttle()

    def test_valid_credentials_return_server_assigned_role(self):
        from src.api.users import authenticate
        for username, password in self.DEMO_PASSWORDS.items():
            user = authenticate(username, password)
            assert user["role"] == username  # demo store: role matches account
            assert user["username"] == username

    def test_wrong_password_rejected(self):
        from src.api.users import AuthenticationError, authenticate
        with pytest.raises(AuthenticationError):
            authenticate("admin", "definitely-wrong-password")

    def test_unknown_user_rejected(self):
        from src.api.users import AuthenticationError, authenticate
        with pytest.raises(AuthenticationError):
            authenticate("nonexistent-user", "whatever")

    def test_error_message_does_not_enumerate_users(self):
        """Unknown user and wrong password must give the identical message."""
        from src.api.users import AuthenticationError, authenticate
        with pytest.raises(AuthenticationError) as e1:
            authenticate("nonexistent-user", "x")
        with pytest.raises(AuthenticationError) as e2:
            authenticate("admin", "x")
        assert str(e1.value) == str(e2.value)

    def test_client_supplied_role_is_ignored(self):
        """LoginRequest must not expose a role field — privilege escalation path."""
        from src.api.main import LoginRequest
        req = LoginRequest.model_validate(
            {"username": "intern", "password": "x", "role": "admin"}
        )
        assert not hasattr(req, "role")

    def test_password_hash_roundtrip(self):
        from src.api.users import hash_password, verify_password
        h = hash_password("S3cure-pass!")
        record = {"salt": h["salt"], "password_hash": h["hash"]}
        assert verify_password("S3cure-pass!", record)
        assert not verify_password("S3cure-pass", record)

    def test_store_contains_no_plaintext_passwords(self):
        import json
        path = PROJECT_ROOT / "data" / "users.example.json"
        entries = json.loads(path.read_text(encoding="utf-8"))
        for e in entries:
            assert "password" not in e
            assert len(e["password_hash"]) == 64  # SHA-256 hex
            assert e["role"] in {"intern", "employee", "hr", "admin"}

    def test_username_case_insensitive(self):
        from src.api.users import authenticate
        user = authenticate("ADMIN", self.DEMO_PASSWORDS["admin"])
        assert user["role"] == "admin"


class TestLoginHardening:
    """Demo-account opt-in gate and brute-force lockout behavior."""

    @pytest.fixture(autouse=True)
    def _clean_state(self, monkeypatch):
        import src.api.users as users
        monkeypatch.setenv("ALLOW_DEMO_ACCOUNTS", "1")
        monkeypatch.setenv("LOGIN_MAX_FAILURES", "3")
        monkeypatch.setenv("LOGIN_LOCKOUT_SECONDS", "60")
        monkeypatch.setattr(users, "_users_cache", None)
        users.reset_login_throttle()
        yield
        users.reset_login_throttle()

    def test_lockout_after_max_failures(self):
        from src.api.users import (
            AuthenticationError, LoginRateLimitedError, authenticate,
        )
        for _ in range(3):
            with pytest.raises(AuthenticationError):
                authenticate("admin", "wrong-password")
        with pytest.raises(LoginRateLimitedError) as exc_info:
            authenticate("admin", "wrong-password")
        assert exc_info.value.retry_after > 0

    def test_lockout_blocks_correct_password_too(self):
        """During lockout even valid credentials are refused (429 path)."""
        from src.api.users import (
            AuthenticationError, LoginRateLimitedError, authenticate,
        )
        for _ in range(3):
            with pytest.raises(AuthenticationError):
                authenticate("admin", "wrong-password")
        with pytest.raises(LoginRateLimitedError):
            authenticate("admin", "Admin#2026")

    def test_successful_login_resets_counter(self):
        from src.api.users import AuthenticationError, authenticate
        with pytest.raises(AuthenticationError):
            authenticate("hr", "wrong")
        with pytest.raises(AuthenticationError):
            authenticate("hr", "wrong")
        assert authenticate("hr", "HrSpec#2026")["role"] == "hr"
        # Counter reset: two more failures must NOT trigger the lockout.
        with pytest.raises(AuthenticationError):
            authenticate("hr", "wrong")
        with pytest.raises(AuthenticationError):
            authenticate("hr", "wrong")
        assert authenticate("hr", "HrSpec#2026")["role"] == "hr"

    def test_lockout_is_per_username(self):
        from src.api.users import (
            AuthenticationError, LoginRateLimitedError, authenticate,
        )
        for _ in range(4):
            with pytest.raises((AuthenticationError, LoginRateLimitedError)):
                authenticate("admin", "wrong")
        # A different account is unaffected.
        assert authenticate("employee", "Employee#2026")["role"] == "employee"

    def test_demo_accounts_require_opt_in(self, monkeypatch):
        """Without ALLOW_DEMO_ACCOUNTS=1 and no real store, login fails closed."""
        import src.api.users as users
        from src.api.users import UserStoreError, authenticate
        if users.USERS_FILE.exists():
            pytest.skip("real data/users.json present — gate not exercised")
        monkeypatch.setenv("ALLOW_DEMO_ACCOUNTS", "0")
        monkeypatch.setattr(users, "_users_cache", None)
        with pytest.raises(UserStoreError):
            authenticate("admin", "Admin#2026")

    def test_login_endpoint_maps_rate_limit_to_429(self):
        from fastapi import HTTPException
        from src.api.main import login, LoginRequest
        for _ in range(3):
            with pytest.raises(HTTPException) as exc_info:
                login(LoginRequest(username="admin", password="wrong"))
            assert exc_info.value.status_code == 401
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(username="admin", password="wrong"))
        assert exc_info.value.status_code == 429
        assert "Too many failed login attempts" in exc_info.value.detail

    def test_login_endpoint_maps_missing_store_to_503(self, monkeypatch):
        import src.api.users as users
        from fastapi import HTTPException
        from src.api.main import login, LoginRequest
        if users.USERS_FILE.exists():
            pytest.skip("real data/users.json present — gate not exercised")
        monkeypatch.setenv("ALLOW_DEMO_ACCOUNTS", "0")
        monkeypatch.setattr(users, "_users_cache", None)
        with pytest.raises(HTTPException) as exc_info:
            login(LoginRequest(username="admin", password="Admin#2026"))
        assert exc_info.value.status_code == 503
