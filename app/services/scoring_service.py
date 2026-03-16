"""Scoring formulas used by memory maintenance and snapshots."""

from __future__ import annotations

import math
from datetime import datetime

from app.core.enums import MemoryStatus
from app.core.time import ensure_utc, utc_now


class ScoringService:
    """Domain scoring helpers."""

    def calculate_decay_score(
        self,
        *,
        base_retention: float,
        last_meaningful_use: datetime | None,
        recall_count: int,
        confidence_score: float,
        now: datetime | None = None,
        decay_lambda: float = 0.05,
    ) -> float:
        """Calculate memory decay based on recency, recall volume, and confidence."""
        current = ensure_utc(now) or utc_now()
        days_since_use = 0.0
        if last_meaningful_use is not None:
            days_since_use = max((current - ensure_utc(last_meaningful_use)).total_seconds() / 86400, 0.0)
        score = base_retention * math.exp(-decay_lambda * days_since_use) * (1 + recall_count * 0.05) * confidence_score
        return max(0.0, min(score, 1.5))

    def status_from_decay(
        self,
        decay_score: float,
        contradiction_risk: float = 0.0,
        superseded: bool = False,
        duplicate: bool = False,
    ) -> str:
        """Map a decay score into a memory status with override hooks."""
        if duplicate:
            return MemoryStatus.DUPLICATE.value
        if superseded:
            return MemoryStatus.SUPERSEDED.value
        if contradiction_risk >= 0.75:
            return MemoryStatus.CONFLICTED.value
        if decay_score >= 0.65:
            return MemoryStatus.ACTIVE.value
        if decay_score >= 0.40:
            return MemoryStatus.WARM.value
        if decay_score >= 0.20:
            return MemoryStatus.STALE.value
        return MemoryStatus.ARCHIVED.value

    def calculate_snapshot_score(
        self,
        *,
        importance_score: float,
        confidence_score: float,
        average_relevance_score: float,
        recall_signal: float,
        freshness_signal: float,
    ) -> float:
        """Calculate the snapshot selection score."""
        return (
            0.35 * importance_score
            + 0.25 * confidence_score
            + 0.20 * average_relevance_score
            + 0.10 * recall_signal
            + 0.10 * freshness_signal
        )
