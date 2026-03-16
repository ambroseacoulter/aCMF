"""Container persistence operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Container


class ContainerRepository:
    """Repository for container records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_existing_ids(self, user_id: str, container_ids: list[str]) -> list[str]:
        """Return existing container IDs for a given user."""
        if not container_ids:
            return []
        stmt = select(Container.id).where(Container.user_id == user_id, Container.id.in_(container_ids))
        return list(self.session.scalars(stmt))

    def create_missing(self, user_id: str, containers: list[dict], fallback_type: str = "generic") -> list[str]:
        """Create any missing containers for the user."""
        container_ids = [container["id"] for container in containers]
        existing = set(self.get_existing_ids(user_id, container_ids))
        created: list[str] = []
        for container in containers:
            if container["id"] in existing:
                continue
            self.session.add(
                Container(
                    id=container["id"],
                    user_id=user_id,
                    container_type=container.get("type") or fallback_type,
                )
            )
            created.append(container["id"])
        self.session.flush()
        return created
