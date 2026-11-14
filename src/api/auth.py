"""JWT authentication utilities for the FastAPI application.

SECRET_KEY must be set via the SECRET_KEY environment variable (never hardcoded).
Tokens expire after 60 minutes.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException, status

ALGORITHM = "HS256"
TOKEN_EXPIRY_MINUTES = 60

VALID_ROLES = frozenset({"intern", "employee", "hr", "admin"})


def _get_secret() -> str:
    key = os.getenv("SECRET_KEY")
    if not key:
        raise RuntimeError(
            "SECRET_KEY is not set. Add it to .env before starting the server."
        )
    return key


def create_token(role: str) -> str:
    """Create a signed JWT token encoding *role* with a 60-minute expiry.

    Parameters
    ----------
    role:
        One of ``intern``, ``employee``, ``hr``, ``admin``.

    Returns
    -------
    str
        Encoded JWT string.

    Raises
    ------
    ValueError
        If *role* is not recognised.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role {role!r}. Valid: {sorted(VALID_ROLES)}")
    now = datetime.now(tz=timezone.utc)
    payload = {
        "role": role,
        "exp": now + timedelta(minutes=TOKEN_EXPIRY_MINUTES),
        "iat": now,
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """Decode and validate *token*.

    Returns
    -------
    str
        The role encoded in the token.

    Raises
    ------
    HTTPException (401)
        If the token is invalid, expired, or tampered with.
    """
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        role: str = payload["role"]
        if role not in VALID_ROLES:
            raise ValueError(f"Token contains unknown role: {role!r}")
        return role
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


def get_current_role(authorization: str = Header(...)) -> str:
    """FastAPI dependency that extracts the role from the Authorization header.

    Expected header format: ``Authorization: Bearer <token>``

    The role is **always** taken from the server-side JWT — it is never read
    from the request body or query parameters.
    """
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be: Bearer <token>",
        )
    return decode_token(token)
