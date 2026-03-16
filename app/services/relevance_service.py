"""Runtime relevance calculations."""

from __future__ import annotations

from datetime import datetime

from app.core.time import ensure_utc, utc_now
from app.db.models import Memory


class RelevanceService:
    """Score and update memory relevance."""

    def calculate_effective_relevance(
        self,
        *,
        query_similarity: float,
        importance_score: float,
        confidence_score: float,
        recall_signal: float,
        recency_signal: float,
        graph_density_signal: float,
        stale_penalty: float = 0.0,
        contradiction_penalty: float = 0.0,
        superseded_penalty: float = 0.0,
    ) -> float:
        """Calculate the effective relevance formula."""
        score = (
            0.40 * query_similarity
            + 0.20 * importance_score
            + 0.15 * confidence_score
            + 0.10 * recall_signal
            + 0.10 * recency_signal
            + 0.05 * graph_density_signal
            - stale_penalty
            - contradiction_penalty
            - superseded_penalty
        )
        return max(0.0, min(score, 1.0))

    def touch_memory(
        self,
        memory: Memory,
        *,
        query_similarity: float,
        graph_density_signal: float = 0.0,
        stale_penalty: float = 0.0,
        contradiction_penalty: float = 0.0,
        superseded_penalty: float = 0.0,
        now: datetime | None = None,
    ) -> Memory:
        """Update a memory after it has been used."""
        current = ensure_utc(now) or utc_now()
        recall_signal = min(memory.recall_count / 20.0, 1.0)
        updated_at = ensure_utc(memory.updated_at)
        if updated_at:
            age_days = max((current - updated_at).total_seconds() / 86400, 0.0)
            recency_signal = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
        else:
            recency_signal = 0.5
        effective = self.calculate_effective_relevance(
            query_similarity=query_similarity,
            importance_score=memory.importance_score,
            confidence_score=memory.confidence_score,
            recall_signal=recall_signal,
            recency_signal=recency_signal,
            graph_density_signal=graph_density_signal,
            stale_penalty=stale_penalty,
            contradiction_penalty=contradiction_penalty,
            superseded_penalty=superseded_penalty,
        )
        memory.recall_count += 1
        memory.last_recalled_at = current
        memory.current_relevance_score = effective
        prior_total = memory.average_relevance_score * max(memory.recall_count - 1, 0)
        memory.average_relevance_score = (prior_total + effective) / max(memory.recall_count, 1)
        return memory
