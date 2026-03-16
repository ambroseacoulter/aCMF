"""Hourly maintenance and snapshot generation orchestration."""

from __future__ import annotations

from datetime import datetime

from app.core.config import Settings
from app.core.enums import MemoryStatus
from app.core.time import ensure_utc, utc_now
from app.engines.graph_engine import GraphEngine
from app.llms.cortex import CortexLLM, CortexReviewToolExecutor
from app.llms.prompting import PromptRenderContext
from app.services.maintenance_service import MaintenanceService
from app.services.memory_service import MemoryService
from app.services.scoring_service import ScoringService
from app.storage.memory_repo import MemoryRepository
from app.storage.snapshot_repo import SnapshotRepository
from app.storage.user_repo import UserRepository


class CortexEngine:
    """Run hourly maintenance and overwrite per-user snapshots."""

    def __init__(
        self,
        *,
        settings: Settings,
        user_repo: UserRepository,
        memory_repo: MemoryRepository,
        snapshot_repo: SnapshotRepository,
        scoring_service: ScoringService,
        maintenance_service: MaintenanceService,
        memory_service: MemoryService,
        graph_engine: GraphEngine,
        cortex_llm: CortexLLM,
    ) -> None:
        self.settings = settings
        self.user_repo = user_repo
        self.memory_repo = memory_repo
        self.snapshot_repo = snapshot_repo
        self.scoring_service = scoring_service
        self.maintenance_service = maintenance_service
        self.memory_service = memory_service
        self.graph_engine = graph_engine
        self.cortex_llm = cortex_llm

    def run_hourly(self, user_id: str) -> list[str]:
        """Run hourly maintenance for a single user."""
        user = self.user_repo.get(user_id)
        if user is None:
            return []
        memories = self.memory_repo.list_user_global_memories(user_id)
        now = utc_now()
        for memory in memories:
            last_use = ensure_utc(memory.last_recalled_at) or ensure_utc(memory.updated_at)
            memory.decay_score = self.scoring_service.calculate_decay_score(
                base_retention=max(memory.importance_score, 0.3),
                last_meaningful_use=last_use,
                recall_count=memory.recall_count,
                confidence_score=memory.confidence_score,
                now=now,
            )
            memory.status = self.scoring_service.status_from_decay(
                memory.decay_score,
                contradiction_risk=memory.contradiction_risk,
                superseded=bool(memory.superseded_by_memory_id),
                duplicate=memory.status == MemoryStatus.DUPLICATE.value,
            )
        graph_result = self.graph_engine.traverse(user_id, "", limit=self.settings.deep_graph_limit)
        contradiction_summaries = self.memory_repo.summarize_open_contradictions(user_id, limit=15)
        lineage_summaries = self.memory_repo.summarize_lineage(user_id, limit=15)
        bundle = self.maintenance_service.build_bundle(user_id, memories)
        context = PromptRenderContext(
            user_id=user_id,
            subject="Hourly cortex maintenance",
            memories=memories,
            graph_facts=graph_result.facts,
            contradiction_summaries=contradiction_summaries,
            lineage_summaries=lineage_summaries,
            budgets={"max_snapshot_memories": 12},
            extras={"snapshot_dirty": user.snapshot_dirty, "analysis_notes": bundle.analysis_notes},
        )
        executor = CortexReviewToolExecutor(user_id=user_id, memory_repo=self.memory_repo, bundle=bundle)
        review_result = self.cortex_llm.review_proposals(context, bundle, executor)
        validation_errors = executor.validate()
        if validation_errors:
            raise ValueError("Invalid cortex review session: {0}".format("; ".join(validation_errors)))
        commit_result = self.memory_service.apply_cortex_review(
            user_id=user_id,
            tool_session=review_result.tool_session,
            bundle=bundle,
        )
        graph_mutation = self.graph_engine.persist_canonical_graph(
            user_id=user_id,
            memory_nodes=[memory.id for memory in memories],
            entities=[],
            relations=[],
            memory_links=[],
            graph_edges=commit_result.graph_edges,
        )
        selected = self._resolve_snapshot_selection(commit_result.snapshot_memory_ids, memories, now)
        summary_context = PromptRenderContext(
            user_id=user_id,
            subject="Hourly user/global snapshot",
            memories=selected,
            graph_facts=graph_result.facts,
            contradiction_summaries=contradiction_summaries,
            lineage_summaries=lineage_summaries,
            budgets={"max_snapshot_memories": 12},
            extras={
                "snapshot_dirty": user.snapshot_dirty,
                "review_reasoning_summary": review_result.reasoning_summary,
                "applied_proposal_ids": commit_result.applied_proposal_ids,
            },
        )
        summary = self.cortex_llm.summarize_snapshot(summary_context)
        self.snapshot_repo.upsert_latest(
            user_id=user_id,
            summary=summary.summary,
            memory_refs=[memory.id for memory in selected],
            health_note=summary.health_note,
        )
        self.user_repo.mark_snapshot_dirty(user_id, False)
        return graph_mutation.projection_event_ids

    def _select_snapshot_memories(self, memories: list, now: datetime) -> list:
        """Choose snapshot memories from user/global healthy memories."""
        ranked = sorted(
            [
                memory
                for memory in memories
                if memory.scope_type in {"user", "global"}
                and memory.status not in {"archived", "duplicate", "superseded"}
            ],
            key=lambda memory: self.scoring_service.calculate_snapshot_score(
                importance_score=memory.importance_score,
                confidence_score=memory.confidence_score,
                average_relevance_score=memory.average_relevance_score,
                recall_signal=min(memory.recall_count / 20.0, 1.0),
                freshness_signal=max(
                    0.0,
                    1.0 - min(((now - ensure_utc(memory.updated_at)).total_seconds() / 86400) / 30.0, 1.0),
                ),
            ),
            reverse=True,
        )
        return ranked[:12]

    def _resolve_snapshot_selection(self, selected_memory_ids: list[str], memories: list, now: datetime) -> list:
        """Resolve reviewed snapshot selection or fall back to programmatic ranking."""
        if selected_memory_ids:
            selected_map = {memory.id: memory for memory in self.memory_repo.get_by_ids(selected_memory_ids)}
            selected = [selected_map[memory_id] for memory_id in selected_memory_ids if memory_id in selected_map]
            filtered = [
                memory
                for memory in selected
                if memory.scope_type in {"user", "global"}
                and memory.status not in {"archived", "duplicate", "superseded"}
            ]
            if filtered:
                return filtered[:12]
        return self._select_snapshot_memories(memories, now)
