"""Read-path orchestration for context and deep-memory requests."""

from __future__ import annotations

import logging
from collections import Counter

from app.api.schemas.common import RetrievalDiagnostics
from app.api.schemas.context import ContextRequest, ContextResponse
from app.api.schemas.deep_memory import DeepMemoryRequest, DeepMemoryResponse, EvidenceItem
from app.core.config import Settings
from app.core.enums import ReadMode, ScopeLevel
from app.db.models import Memory
from app.engines.graph_engine import GraphEngine
from app.llms.context_enhancer import ContextEnhancerLLM, render_context_xml
from app.llms.prompting import PromptRenderContext
from app.retrieval.query_relevance import RetrievalCandidate, graph_fact_score, memory_text, text_overlap_score
from app.retrieval.reranker import Reranker
from app.retrieval.scope_resolver import ScopeResolver
from app.retrieval.vector_search import VectorSearch, VectorSearchHit
from app.services.memory_service import MemoryService
from app.storage.container_repo import ContainerRepository
from app.storage.memory_repo import MemoryRepository
from app.storage.snapshot_repo import SnapshotRepository
from app.storage.user_repo import UserRepository

logger = logging.getLogger(__name__)


class ContextEngine:
    """Build prompt-ready context and grounded deep-memory answers."""

    def __init__(
        self,
        *,
        settings: Settings,
        user_repo: UserRepository,
        container_repo: ContainerRepository,
        memory_repo: MemoryRepository,
        snapshot_repo: SnapshotRepository,
        vector_search: VectorSearch,
        scope_resolver: ScopeResolver,
        reranker: Reranker,
        memory_service: MemoryService,
        graph_engine: GraphEngine,
        enhancer: ContextEnhancerLLM,
    ) -> None:
        self.settings = settings
        self.user_repo = user_repo
        self.container_repo = container_repo
        self.memory_repo = memory_repo
        self.snapshot_repo = snapshot_repo
        self.vector_search = vector_search
        self.scope_resolver = scope_resolver
        self.reranker = reranker
        self.memory_service = memory_service
        self.graph_engine = graph_engine
        self.enhancer = enhancer

    def build_context(self, request: ContextRequest) -> ContextResponse:
        """Build the `/v1/context` response."""
        diagnostics, candidates, graph_result, context = self._prepare_read(
            user_id=request.user_id,
            query=request.message,
            containers=[container.id for container in request.containers],
            scope_level=request.scope_level,
            read_mode=request.read_mode,
            budgets=request.budgets.model_dump(),
            metadata=request.metadata.model_dump(),
        )
        if not diagnostics.user_found:
            return ContextResponse(
                status="not_found",
                has_usable_context=False,
                context_enhancement="",
                abstained_reason="Unknown user.",
                diagnostics=diagnostics,
            )
        if not candidates:
            return ContextResponse(
                status="ok",
                has_usable_context=False,
                context_enhancement="",
                abstained_reason="No relevant grounded memory was found for this query.",
                diagnostics=diagnostics,
            )
        payload = self.enhancer.synthesize_context(request.scope_level, context)
        used_memories = self._select_used_memories(candidates, payload.used_memory_ids)
        self.memory_service.touch_memories(
            used_memories,
            query_similarity_map={memory.id: 0.75 for memory in used_memories},
            graph_density_signal=graph_result.graph_density_signal,
        )
        diagnostics.used_memory_count = len(used_memories)
        diagnostics.evidence_strength = self._evidence_strength(used_memories)
        if not payload.has_usable_context:
            payload.abstained_reason = payload.abstained_reason or "No sufficiently grounded context found."
        enhancement = render_context_xml(diagnostics.scope_applied, payload) if payload.has_usable_context else ""
        return ContextResponse(
            status="ok",
            has_usable_context=payload.has_usable_context,
            context_enhancement=enhancement,
            abstained_reason=payload.abstained_reason,
            diagnostics=diagnostics,
        )

    def answer_deep_memory(self, request: DeepMemoryRequest) -> DeepMemoryResponse:
        """Build the `/v1/deep-memory` response."""
        diagnostics, candidates, graph_result, context = self._prepare_read(
            user_id=request.user_id,
            query=request.query,
            containers=[container.id for container in request.containers],
            scope_level=request.scope_level,
            read_mode=request.read_mode,
            budgets=request.budgets.model_dump(),
            metadata=request.metadata.model_dump(),
        )
        if not diagnostics.user_found:
            return DeepMemoryResponse(
                status="not_found",
                answer="Unknown user.",
                abstained=True,
                abstained_reason="Unknown user.",
                used_memory_count=0,
                diagnostics=diagnostics,
                evidence=[],
            )
        if not candidates:
            return DeepMemoryResponse(
                status="ok",
                answer="I do not have enough grounded memory evidence to answer that confidently.",
                abstained=True,
                abstained_reason="No relevant grounded memory was found for this query.",
                used_memory_count=0,
                diagnostics=diagnostics,
                evidence=[],
            )
        payload = self.enhancer.answer_deep_memory(context)
        used_memories = self._select_used_memories(candidates, payload.used_memory_ids)
        self.memory_service.touch_memories(
            used_memories,
            query_similarity_map={memory.id: 0.8 for memory in used_memories},
            graph_density_signal=graph_result.graph_density_signal,
        )
        diagnostics.used_memory_count = len(used_memories)
        diagnostics.evidence_strength = self._evidence_strength(used_memories)
        abstained = payload.abstained or diagnostics.evidence_strength < self.settings.evidence_abstain_threshold
        abstained_reason = payload.abstained_reason
        if abstained and not abstained_reason:
            abstained_reason = "Evidence strength was too low for a grounded answer."
        evidence = [
            EvidenceItem(
                memory_id=memory.id,
                scope_type=memory.scope_type,
                bucket_id=memory.bucket_id,
                relevance=memory.current_relevance_score,
                support=max(memory.confidence_score, memory.average_relevance_score),
            )
            for memory in used_memories
        ]
        answer = payload.answer
        if abstained and diagnostics.evidence_strength < self.settings.evidence_abstain_threshold:
            answer = "I do not have enough grounded memory evidence to answer that confidently."
        return DeepMemoryResponse(
            status="ok",
            answer=answer,
            abstained=abstained,
            abstained_reason=abstained_reason,
            used_memory_count=len(used_memories),
            diagnostics=diagnostics,
            evidence=evidence,
        )

    def _prepare_read(
        self,
        *,
        user_id: str,
        query: str,
        containers: list[str],
        scope_level: ScopeLevel,
        read_mode: ReadMode,
        budgets: dict,
        metadata: dict,
    ) -> tuple[RetrievalDiagnostics, list[Memory], object, PromptRenderContext]:
        """Resolve scope, retrieve candidates, and build a prompt render context."""
        user = self.user_repo.get(user_id)
        if user is None:
            diagnostics = RetrievalDiagnostics(
                scope_applied=scope_level,
                read_mode=read_mode,
                user_found=False,
                candidate_count=0,
                used_memory_count=0,
                missing_containers=containers,
                source_breakdown={},
                evidence_strength=0.0,
                warnings=["user_not_found"],
            )
            return diagnostics, [], type("GraphResult", (), {"facts": [], "graph_density_signal": 0.0})(), PromptRenderContext(
                user_id=user_id,
                subject=query,
                memories=[],
                graph_facts=[],
                contradiction_summaries=[],
                lineage_summaries=[],
                budgets=budgets,
                extras=metadata,
            )
        existing_containers = self.container_repo.get_existing_ids(user_id, containers)
        resolved = self.scope_resolver.resolve(scope_level, containers, existing_containers)
        profile = self.settings.read_profile()[read_mode.value]
        limit = min(budgets.get("max_candidate_memories", profile["candidate_limit"]), profile["candidate_limit"])
        snapshot_ids = self._snapshot_memory_ids(user_id, resolved.applied_scope)
        snapshot_memories = self.memory_repo.get_by_ids(snapshot_ids)
        vector_hits = self.vector_search.search_with_scores(
            user_id=user_id,
            scope_types=resolved.scope_types,
            bucket_ids=resolved.container_ids,
            query=query,
            limit=limit,
        )
        metadata_memories = self.memory_repo.list_high_signal_candidates(
            user_id=user_id,
            scope_types=resolved.scope_types,
            bucket_ids=resolved.container_ids,
            limit=max(5, limit // 2),
        )
        raw_sources = {
            "snapshot": len(snapshot_memories),
            "vector": len(vector_hits),
            "metadata": len(metadata_memories),
        }
        initial_candidates = self._merge_candidates(query, snapshot_memories, vector_hits, metadata_memories)
        relevant_candidates = self._filter_relevant_candidates(initial_candidates)
        initial_relevant_sources = self._source_breakdown(relevant_candidates, {candidate.memory.id for candidate in relevant_candidates})
        if relevant_candidates:
            graph_result = self.graph_engine.traverse(
                user_id,
                query,
                limit=profile["graph_limit"],
                seed_memory_ids=[candidate.memory.id for candidate in relevant_candidates],
            )
            graph_memories = self.memory_repo.get_by_ids(graph_result.memory_ids)
            graph_candidates = self._merge_graph_candidates(query, graph_memories, graph_result.facts)
        else:
            graph_result = type("GraphResult", (), {"facts": [], "memory_ids": [], "graph_density_signal": 0.0})()
            graph_memories = []
            graph_candidates = []
        raw_sources["graph_memories"] = len(graph_memories)
        raw_sources["graph"] = len(graph_result.facts)
        final_candidates = self._filter_relevant_candidates(relevant_candidates + graph_candidates)
        candidates = self.reranker.rerank(
            [candidate.memory for candidate in final_candidates],
            limit,
            relevance_scores={candidate.memory.id: candidate.query_relevance for candidate in final_candidates},
        )
        candidate_sources = self._source_breakdown(final_candidates, {memory.id for memory in candidates})
        contradiction_summaries = self.memory_repo.summarize_open_contradictions(user_id, limit=8)
        lineage_summaries = self.memory_repo.summarize_lineage(user_id, limit=8)
        logger.info(
            "Read retrieval user_id=%s query=%r raw_sources=%s initial_relevant_sources=%s final_relevant_sources=%s final_prompt_candidates=%s",
            user_id,
            query,
            raw_sources,
            initial_relevant_sources,
            candidate_sources,
            len(candidates),
        )
        diagnostics = RetrievalDiagnostics(
            scope_applied=resolved.applied_scope,
            read_mode=read_mode,
            user_found=True,
            candidate_count=len(candidates),
            used_memory_count=0,
            missing_containers=resolved.missing_containers,
            source_breakdown=candidate_sources,
            evidence_strength=self._evidence_strength(candidates),
            warnings=["missing_containers"] if resolved.missing_containers else [],
        )
        context = PromptRenderContext(
            user_id=user_id,
            subject=query,
            memories=candidates,
            graph_facts=graph_result.facts,
            contradiction_summaries=contradiction_summaries,
            lineage_summaries=lineage_summaries,
            budgets=budgets,
            extras=metadata,
        )
        return diagnostics, candidates, graph_result, context

    def _snapshot_memory_ids(self, user_id: str, scope_level: ScopeLevel) -> list[str]:
        """Return snapshot memory refs when the read path includes user/global scope."""
        if scope_level == ScopeLevel.USER:
            return []
        snapshot = self.snapshot_repo.get_latest(user_id)
        return snapshot.memory_refs if snapshot is not None else []

    def _merge_candidates(
        self,
        query: str,
        snapshot_memories: list[Memory],
        vector_hits: list[VectorSearchHit],
        metadata_memories: list[Memory],
    ) -> list[RetrievalCandidate]:
        """Merge raw scoped retrieval into query-scored candidates."""
        candidates: dict[str, RetrievalCandidate] = {}
        for memory in snapshot_memories:
            candidate = candidates.setdefault(memory.id, RetrievalCandidate(memory=memory))
            candidate.sources.add("snapshot")
        for hit in vector_hits:
            candidate = candidates.setdefault(hit.memory.id, RetrievalCandidate(memory=hit.memory))
            candidate.sources.add("vector")
            candidate.vector_similarity = max(candidate.vector_similarity, hit.similarity)
        for memory in metadata_memories:
            candidate = candidates.setdefault(memory.id, RetrievalCandidate(memory=memory))
            candidate.sources.add("metadata")
        for candidate in candidates.values():
            candidate.lexical_similarity = self._text_relevance(query, candidate.memory)
            candidate.query_relevance = self._candidate_relevance(candidate)
        return list(candidates.values())

    def _merge_graph_candidates(
        self,
        query: str,
        graph_memories: list[Memory],
        graph_facts: list,
    ) -> list[RetrievalCandidate]:
        """Merge graph expansion results into query-scored candidates."""
        graph_score = graph_fact_score(query, graph_facts)
        candidates: list[RetrievalCandidate] = []
        for memory in graph_memories:
            lexical_similarity = self._text_relevance(query, memory)
            candidate = RetrievalCandidate(
                memory=memory,
                sources={"graph_memories"},
                lexical_similarity=lexical_similarity,
                graph_similarity=max(graph_score, lexical_similarity),
            )
            candidate.query_relevance = self._candidate_relevance(candidate, is_graph=True)
            candidates.append(candidate)
        return candidates

    def _filter_relevant_candidates(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Drop candidates that do not meet query-relevance thresholds."""
        relevant: dict[str, RetrievalCandidate] = {}
        for candidate in candidates:
            if not self._passes_relevance_gate(candidate):
                continue
            existing = relevant.get(candidate.memory.id)
            if existing is None or candidate.query_relevance > existing.query_relevance:
                relevant[candidate.memory.id] = candidate
                continue
            existing.sources.update(candidate.sources)
            existing.vector_similarity = max(existing.vector_similarity, candidate.vector_similarity)
            existing.lexical_similarity = max(existing.lexical_similarity, candidate.lexical_similarity)
            existing.graph_similarity = max(existing.graph_similarity, candidate.graph_similarity)
            existing.query_relevance = max(existing.query_relevance, candidate.query_relevance)
        return list(relevant.values())

    def _passes_relevance_gate(self, candidate: RetrievalCandidate) -> bool:
        """Decide whether a candidate is sufficiently relevant to the query."""
        if "graph_memories" in candidate.sources:
            return (
                candidate.graph_similarity >= self.settings.graph_query_relevance_threshold
                or candidate.lexical_similarity >= self.settings.graph_query_relevance_threshold
            )
        return (
            candidate.lexical_similarity >= self.settings.query_text_relevance_threshold
            or candidate.vector_similarity >= self.settings.query_vector_similarity_threshold
        )

    def _candidate_relevance(self, candidate: RetrievalCandidate, *, is_graph: bool = False) -> float:
        """Blend query-signal inputs into one ranking score."""
        if is_graph:
            return max(candidate.graph_similarity, candidate.lexical_similarity)
        if candidate.lexical_similarity > 0:
            return max(candidate.lexical_similarity, candidate.vector_similarity)
        return candidate.vector_similarity

    @staticmethod
    def _text_relevance(query: str, memory: Memory) -> float:
        """Estimate lexical relevance between the query and memory text."""
        return text_overlap_score(query, memory_text(memory))

    @staticmethod
    def _source_breakdown(candidates: list[RetrievalCandidate], allowed_memory_ids: set[str]) -> dict[str, int]:
        """Return relevant-only source counts for selected candidates."""
        counts: Counter[str] = Counter()
        for candidate in candidates:
            if candidate.memory.id not in allowed_memory_ids:
                continue
            for source in candidate.sources:
                counts[source] += 1
        return dict(counts)

    @staticmethod
    def _select_used_memories(candidates: list[Memory], used_memory_ids: list[str]) -> list[Memory]:
        """Select the memories actually used in synthesis."""
        if not candidates:
            return []
        if not used_memory_ids:
            return candidates[: min(5, len(candidates))]
        allowed = set(used_memory_ids)
        selected = [memory for memory in candidates if memory.id in allowed]
        return selected[: min(10, len(selected))]

    @staticmethod
    def _evidence_strength(memories: list[Memory]) -> float:
        """Estimate evidence strength from selected memories."""
        if not memories:
            return 0.0
        score = sum((memory.confidence_score + memory.current_relevance_score + memory.average_relevance_score) / 3.0 for memory in memories)
        return max(0.0, min(score / len(memories), 1.0))
