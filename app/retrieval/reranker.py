"""Candidate reranking helpers."""

from __future__ import annotations

from collections.abc import Iterable

from app.db.models import Memory


class Reranker:
    """Deterministic blended candidate reranker."""

    def rerank(self, memories: Iterable[Memory], limit: int, relevance_scores: dict[str, float] | None = None) -> list[Memory]:
        """Order memories using score fields and lifecycle penalties."""
        relevance_scores = relevance_scores or {}
        ranked = sorted(
            memories,
            key=lambda memory: (
                relevance_scores.get(memory.id, 0.0) * 2.0
                + memory.current_relevance_score
                + memory.average_relevance_score
                + memory.importance_score
                + memory.confidence_score
                - memory.contradiction_risk
                - (0.2 if memory.superseded_by_memory_id else 0.0)
                - (0.15 if memory.status in {"duplicate", "archived", "superseded"} else 0.0)
            ),
            reverse=True,
        )
        deduped: list[Memory] = []
        seen: set[str] = set()
        for memory in ranked:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            deduped.append(memory)
            if len(deduped) >= limit:
                break
        return deduped
