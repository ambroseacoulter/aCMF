"""Programmatic maintenance proposal generation for Cortex."""

from __future__ import annotations

from difflib import SequenceMatcher

from app.core.enums import MemoryStatus
from app.db.models import Memory
from app.services.scoring_service import ScoringService
from app.services.tool_session_service import MaintenanceProposal, MaintenanceProposalBundle


class MaintenanceService:
    """Compute programmatic maintenance proposals per user."""

    def __init__(self, scoring_service: ScoringService) -> None:
        self.scoring_service = scoring_service

    def build_bundle(self, user_id: str, memories: list[Memory]) -> MaintenanceProposalBundle:
        """Build a maintenance proposal bundle."""
        proposals: list[MaintenanceProposal] = []
        analysis_notes: list[str] = []
        proposal_counter = 1

        for memory in memories:
            suggested_status = self.scoring_service.status_from_decay(
                memory.decay_score,
                contradiction_risk=memory.contradiction_risk,
                superseded=bool(memory.superseded_by_memory_id),
                duplicate=memory.status == MemoryStatus.DUPLICATE.value,
            )
            if suggested_status != memory.status:
                proposals.append(
                    MaintenanceProposal(
                        proposal_id="proposal_{0}".format(proposal_counter),
                        proposal_type="status_update",
                        memory_ids=[memory.id],
                        status=suggested_status,
                        archived_reason="decay_threshold" if suggested_status == MemoryStatus.ARCHIVED.value else None,
                        confidence_score=0.75,
                        rationale="Programmatic decay/status evaluation suggested a status change.",
                    )
                )
                proposal_counter += 1

        for idx, left in enumerate(memories):
            for right in memories[idx + 1 :]:
                similarity = self._text_similarity(left.summary or left.content, right.summary or right.content)
                if similarity >= 0.92 and left.memory_type == right.memory_type and left.scope_type == right.scope_type:
                    proposals.append(
                        MaintenanceProposal(
                            proposal_id="proposal_{0}".format(proposal_counter),
                            proposal_type="duplicate_candidate",
                            memory_ids=[left.id, right.id],
                            target_memory_id=left.id,
                            confidence_score=similarity,
                            rationale="High textual similarity with matching type and scope.",
                        )
                    )
                    proposal_counter += 1
                elif similarity >= 0.68 and self._looks_contradictory(left.content, right.content):
                    proposals.append(
                        MaintenanceProposal(
                            proposal_id="proposal_{0}".format(proposal_counter),
                            proposal_type="contradiction_candidate",
                            memory_ids=[left.id, right.id],
                            topic=self._topic_from_memories(left, right),
                            confidence_score=0.6,
                            rationale="Potential contradiction detected from overlapping topic with opposing language.",
                        )
                    )
                    proposal_counter += 1

        snapshot_candidates = [
            memory.id
            for memory in sorted(
                [
                    memory
                    for memory in memories
                    if memory.scope_type in {"user", "global"}
                    and memory.status not in {"archived", "duplicate", "superseded"}
                ],
                key=lambda memory: (
                    memory.average_relevance_score,
                    memory.importance_score,
                    memory.confidence_score,
                ),
                reverse=True,
            )[:12]
        ]
        analysis_notes.append("Generated {0} maintenance proposals.".format(len(proposals)))
        return MaintenanceProposalBundle(
            user_id=user_id,
            proposals=proposals,
            snapshot_candidates=snapshot_candidates,
            analysis_notes=analysis_notes,
        )

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        """Return a normalized textual similarity score."""
        return SequenceMatcher(None, left.lower(), right.lower()).ratio()

    @staticmethod
    def _looks_contradictory(left: str, right: str) -> bool:
        """Return whether two texts appear contradictory by simple heuristics."""
        left_text = left.lower()
        right_text = right.lower()
        contradiction_markers = [(" is ", " is not "), (" prefers ", " does not prefer "), (" can ", " cannot ")]
        for positive, negative in contradiction_markers:
            if positive in left_text and negative in right_text:
                return True
            if positive in right_text and negative in left_text:
                return True
        return False

    @staticmethod
    def _topic_from_memories(left: Memory, right: Memory) -> str:
        """Derive a short contradiction topic."""
        left_topic = (left.summary or left.content).split(".")[0][:80]
        right_topic = (right.summary or right.content).split(".")[0][:80]
        return "{0} / {1}".format(left_topic, right_topic)
