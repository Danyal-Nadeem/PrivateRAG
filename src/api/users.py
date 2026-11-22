"""Credential-based user store with server-assigned roles.

Replaces the passwordless /login-role SSO stub: a client can no longer
self-assign a privilege level. Roles are looked up server-side from the
user store after successful password verification.

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes (stdlib only —
no new dependency). Verification uses hmac.compare_digest (constant time).

Store location:
- data/users.json        — real store (gitignored, create from the example)
- data/users.example.json — shipped DEMO accounts for the course prototype

The demo store is gated: it is only used when ALLOW_DEMO_ACCOUNTS=1 is set
explicitly. Without a real users.json and without the opt-in flag, login
fails closed (UserStoreError) instead of silently accepting the published
demo credentials.

Brute-force defense: repeated failed attempts for a username trigger a
temporary in-memory lockout (HTTP 429 via the API). Tunables:
LOGIN_MAX_FAILURES (default 5) and LOGIN_LOCKOUT_SECONDS (default 300).

This is still a prototype identity store, not a production IdP — but
authentication now precedes authorization, as the IS design requires.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("privaterag.auth")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
USERS_FILE = PROJECT_ROOT / "data" / "users.json"
EXAMPLE_USERS_FILE = PROJECT_ROOT / "data" / "users.example.json"

PBKDF2_ITERATIONS = 200_000


class AuthenticationError(Exception):
    """Raised when username/password verification fails.

    The message is intentionally generic so responses do not reveal
    whether the username exists (user-enumeration defense).
    """


class UserStoreError(RuntimeError):
    """Raised when no usable user store is configured.

    Login must fail closed in this state: without a real store (or the
    explicit ALLOW_DEMO_ACCOUNTS=1 opt-in) the server cannot authenticate
    anyone and must not fall back to published demo credentials.
    """


class LoginRateLimitedError(Exception):
    """Raised when a username is temporarily locked after repeated failures."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(
            f"Too many failed login attempts. Try again in {retry_after} seconds."
        )


def hash_password(password: str, salt: str | None = None) -> Dict[str, str]:
    """Return {"salt", "hash"} for *password* using PBKDF2-HMAC-SHA256."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return {"salt": salt, "hash": digest.hex()}


def verify_password(password: str, record: Dict[str, Any]) -> bool:
    """Constant-time verification of *password* against a stored record."""
    try:
        expected = bytes.fromhex(record["password_hash"])
        salt = bytes.fromhex(record["salt"])
    except (KeyError, ValueError):
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate, expected)


_users_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _load_users() -> Dict[str, Dict[str, Any]]:
    global _users_cache
    if _users_cache is not None:
        return _users_cache

    path = USERS_FILE
    if not path.exists():
        if os.getenv("ALLOW_DEMO_ACCOUNTS", "0") != "1":
            logger.error(
                "No user store at %s and demo accounts are NOT enabled "
                "(ALLOW_DEMO_ACCOUNTS != 1). Refusing to authenticate anyone.",
                USERS_FILE,
            )
            raise UserStoreError(
                "No user store configured. Create data/users.json, or set "
                "ALLOW_DEMO_ACCOUNTS=1 to opt in to the published demo accounts."
            )
        path = EXAMPLE_USERS_FILE
        logger.warning(
            "data/users.json not found — using shipped DEMO accounts from %s "
            "(ALLOW_DEMO_ACCOUNTS=1). Demo credentials are public: create a "
            "real store before any non-prototype use.", path,
        )
    if not path.exists():
        raise UserStoreError(
            "No user store found. Copy data/users.example.json to data/users.json."
        )

    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    users: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        username = e["username"].strip().lower()
        users[username] = {
            "username": username,
            "role": e["role"],
            "salt": e["salt"],
            "password_hash": e["password_hash"],
        }
    _users_cache = users
    return users


# ---------------------------------------------------------------------------
# Brute-force lockout (in-memory, per username)
#
# Prototype scope: state lives in this process only (no Redis), so it resets
# on restart and does not span multiple workers. It still makes online
# password guessing against known usernames impractical.
# ---------------------------------------------------------------------------

_login_failures: Dict[str, Dict[str, float]] = {}


def _max_failures() -> int:
    try:
        return max(1, int(os.getenv("LOGIN_MAX_FAILURES", "5")))
    except ValueError:
        return 5


def _lockout_seconds() -> int:
    try:
        return max(1, int(os.getenv("LOGIN_LOCKOUT_SECONDS", "300")))
    except ValueError:
        return 300


def reset_login_throttle() -> None:
    """Clear all lockout state (used by tests)."""
    _login_failures.clear()


def _check_throttle(key: str) -> None:
    entry = _login_failures.get(key)
    if entry is None:
        return
    remaining = entry.get("locked_until", 0.0) - time.monotonic()
    if remaining > 0:
        raise LoginRateLimitedError(int(remaining) + 1)
    if remaining <= 0 and entry.get("locked_until"):
        # Lockout expired — start with a clean slate.
        _login_failures.pop(key, None)


def _record_failure(key: str) -> None:
    entry = _login_failures.setdefault(key, {"count": 0.0, "locked_until": 0.0})
    entry["count"] += 1
    if entry["count"] >= _max_failures():
        seconds = _lockout_seconds()
        entry["locked_until"] = time.monotonic() + seconds
        logger.warning(
            "LOGIN LOCKOUT: username %r locked for %ss after %d failed attempts",
            key, seconds, int(entry["count"]),
        )


def _clear_failures(key: str) -> None:
    _login_failures.pop(key, None)


def authenticate(username: str, password: str) -> Dict[str, str]:
    """Verify credentials and return {"username", "role"} from the store.

    The role is ALWAYS taken from the server-side user record — never
    from client input.

    Raises
    ------
    LoginRateLimitedError
        The username is temporarily locked after repeated failed attempts.
    UserStoreError
        No user store is configured (login fails closed).
    AuthenticationError
        Credentials are invalid (generic message, no user enumeration).
    """
    key = username.strip().lower()
    _check_throttle(key)

    users = _load_users()
    record = users.get(key)

    # Verify even for unknown users (dummy hash) to keep timing uniform
    # and avoid username enumeration.
    if record is None:
        dummy = {"salt": "00" * 16, "hash": "00" * 32}
        verify_password(password, dummy)
        _record_failure(key)
        logger.warning("LOGIN FAILED: unknown username %r", username)
        raise AuthenticationError("Invalid username or password.")

    if not verify_password(password, record):
        _record_failure(key)
        logger.warning("LOGIN FAILED: bad password for user %r", record["username"])
        raise AuthenticationError("Invalid username or password.")

    _clear_failures(key)
    logger.info("LOGIN OK: user=%s role=%s", record["username"], record["role"])
    return {"username": record["username"], "role": record["role"]}