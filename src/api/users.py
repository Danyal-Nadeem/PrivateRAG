"""Credential-based user store with server-assigned roles.

Replaces the passwordless /login-role SSO stub: a client can no longer
self-assign a privilege level. Roles are looked up server-side from the
user store after successful password verification.

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes (stdlib only —
no new dependency). Verification uses hmac.compare_digest (constant time).

Store location:
- data/users.json        — real store (gitignored, create from the example)
- data/users.example.json — shipped DEMO accounts for the course prototype

This is still a prototype identity store, not a production IdP — but
authentication now precedes authorization, as the IS design requires.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
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
        path = EXAMPLE_USERS_FILE
        logger.warning(
            "data/users.json not found — using shipped DEMO accounts from %s. "
            "Create a real store before any non-prototype use.", path,
        )
    if not path.exists():
        raise RuntimeError(
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


def authenticate(username: str, password: str) -> Dict[str, str]:
    """Verify credentials and return {"username", "role"} from the store.

    The role is ALWAYS taken from the server-side user record — never
    from client input. Raises AuthenticationError on any failure.
    """
    users = _load_users()
    record = users.get(username.strip().lower())

    # Verify even for unknown users (dummy hash) to keep timing uniform
    # and avoid username enumeration.
    if record is None:
        dummy = {"salt": "00" * 16, "hash": "00" * 32}
        verify_password(password, dummy)
        logger.warning("LOGIN FAILED: unknown username %r", username)
        raise AuthenticationError("Invalid username or password.")

    if not verify_password(password, record):
        logger.warning("LOGIN FAILED: bad password for user %r", record["username"])
        raise AuthenticationError("Invalid username or password.")

    logger.info("LOGIN OK: user=%s role=%s", record["username"], record["role"])
    return {"username": record["username"], "role": record["role"]}