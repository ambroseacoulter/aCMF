"""Identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_uuid() -> str:
    """Return a new UUID string."""
    return str(uuid4())
