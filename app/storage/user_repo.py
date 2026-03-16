"""User persistence operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User


class UserRepository:
    """Repository for `User` records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, user_id: str) -> User | None:
        """Return a user by ID."""
        return self.session.get(User, user_id)

    def get_all_ids(self) -> list[str]:
        """Return all known user IDs."""
        return list(self.session.scalars(select(User.id)))

    def create(self, user_id: str) -> User:
        """Create a new user."""
        user = User(id=user_id)
        self.session.add(user)
        self.session.flush()
        return user

    def get_or_create(self, user_id: str) -> tuple[User, bool]:
        """Fetch or create a user."""
        user = self.get(user_id)
        if user is not None:
            return user, False
        return self.create(user_id), True

    def mark_snapshot_dirty(self, user_id: str, dirty: bool = True) -> None:
        """Update the snapshot dirty flag."""
        user = self.get(user_id)
        if user is not None:
            user.snapshot_dirty = dirty
