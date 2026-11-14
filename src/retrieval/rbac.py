"""RBAC role-to-access-level mapping and Qdrant filter builder."""
from __future__ import annotations

from qdrant_client import models

ROLE_ACCESS_MAP: dict[str, list[str]] = {
    "intern":   ["public"],
    "employee": ["public", "internal"],
    "hr":       ["public", "internal", "hr_restricted"],
    "admin":    ["public", "internal", "hr_restricted", "exec_restricted"],
}

VALID_ROLES = frozenset(ROLE_ACCESS_MAP.keys())

LEVEL_ORDER: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "hr_restricted": 2,
    "exec_restricted": 3,
}


def get_allowed_levels(role: str) -> list[str]:
    """Return the list of access levels allowed for *role*.

    Raises
    ------
    ValueError
        If *role* is not recognised.
    """
    if role not in ROLE_ACCESS_MAP:
        raise ValueError(
            f"Unknown role {role!r}. Valid roles: {sorted(VALID_ROLES)}"
        )
    return ROLE_ACCESS_MAP[role]


def build_rbac_filter(role: str) -> models.Filter:
    """Build a Qdrant payload filter restricting results to levels allowed for *role*.

    The filter is applied **at query time inside Qdrant** — post-filtering is
    never used.

    Parameters
    ----------
    role:
        One of ``intern``, ``employee``, ``hr``, ``admin``.

    Returns
    -------
    models.Filter
        A Qdrant ``Filter`` with a ``MatchAny`` condition on the
        ``access_level`` payload field.

    Raises
    ------
    ValueError
        If *role* is not recognised.
    """
    allowed = get_allowed_levels(role)
    return models.Filter(
        must=[
            models.FieldCondition(
                key="access_level",
                match=models.MatchAny(any=allowed),
            )
        ]
    )


def is_leak(retrieved_level: str, role: str) -> bool:
    """Return True if *retrieved_level* is above what *role* is allowed to see."""
    allowed = get_allowed_levels(role)
    max_allowed = max(LEVEL_ORDER[lvl] for lvl in allowed)
    return LEVEL_ORDER.get(retrieved_level, 0) > max_allowed
