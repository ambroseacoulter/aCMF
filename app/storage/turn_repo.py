"""Turn-record persistence operations."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import TurnRecord


class TurnRepository:
    """Repository for normalized turn records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: dict) -> TurnRecord:
        """Create a turn record."""
        record = TurnRecord(**payload)
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, turn_id: str) -> TurnRecord | None:
        """Get a turn record by id."""
        return self.session.get(TurnRecord, turn_id)

    def mark_processed(self, turn_id: str, notes: dict | None = None) -> TurnRecord | None:
        """Mark a turn as processed."""
        record = self.get(turn_id)
        if record is not None:
            record.processing_status = "processed"
            record.processing_notes = notes or {}
        return record

    def mark_failed(self, turn_id: str, error_message: str) -> TurnRecord | None:
        """Mark a turn as failed."""
        record = self.get(turn_id)
        if record is not None:
            record.processing_status = "failed"
            record.processing_notes = {"error": error_message}
        return record
