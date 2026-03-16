"""Cortex role integration."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llms.client import LLMClient, ToolDefinition, load_prompt
from app.llms.prompting import PromptRenderContext, format_graph_facts, format_memories, format_string_list
from app.services.tool_session_service import (
    MaintenanceProposalBundle,
    SnapshotCandidateSet,
    StagedContradictionUpdate,
    StagedLineageEvent,
    StagedStatusUpdate,
    ToolSession,
    ToolSessionValidator,
)


class CortexSnapshotSummary(BaseModel):
    """Final snapshot summary payload."""

    summary: str
    health_note: str | None = None


@dataclass
class MaintenanceReviewResult:
    """Result from the cortex maintenance review loop."""

    reasoning_summary: str
    tool_session: ToolSession
    tool_call_count: int


class CortexReviewToolExecutor:
    """Review tools available to Cortex over programmatic proposals."""

    def __init__(self, *, user_id: str, memory_repo, bundle: MaintenanceProposalBundle) -> None:
        self.user_id = user_id
        self.memory_repo = memory_repo
        self.bundle = bundle
        self.session = ToolSession(session_id="cortex_{0}".format(user_id), kind="cortex", user_id=user_id)
        self.validator = ToolSessionValidator()
        self.proposal_index = {proposal.proposal_id: proposal for proposal in bundle.proposals}

    def tools(self) -> list[ToolDefinition]:
        """Return available cortex review tools."""
        return [
            ToolDefinition(
                name="inspect_proposal",
                description="Inspect one programmatic maintenance proposal in detail.",
                parameters={
                    "type": "object",
                    "properties": {"proposal_id": {"type": "string"}},
                    "required": ["proposal_id"],
                },
            ),
            ToolDefinition(
                name="inspect_memory",
                description="Inspect one memory by id.",
                parameters={
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            ),
            ToolDefinition(
                name="stage_status_update",
                description="Stage a status update for a memory.",
                parameters=StagedStatusUpdate.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_lineage_event",
                description="Stage a lineage event between two memories.",
                parameters=StagedLineageEvent.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_contradiction_update",
                description="Stage a contradiction update.",
                parameters=StagedContradictionUpdate.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_snapshot_selection_override",
                description="Override the selected snapshot memory ids.",
                parameters=SnapshotCandidateSet.model_json_schema(),
            ),
            ToolDefinition(
                name="finalize_cortex_review",
                description="Finalize maintenance review once staged decisions are complete.",
                parameters={
                    "type": "object",
                    "properties": {"reasoning_summary": {"type": "string"}},
                    "required": ["reasoning_summary"],
                },
            ),
        ]

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Execute one cortex tool call."""
        if name == "inspect_proposal":
            proposal = self.proposal_index.get(str(arguments["proposal_id"]))
            return {"proposal": proposal.model_dump(mode="json") if proposal else None}
        if name == "inspect_memory":
            memory = self.memory_repo.get(str(arguments["memory_id"]))
            if memory is None:
                return {"memory": None}
            return {
                "memory": {
                    "memory_id": memory.id,
                    "summary": memory.summary,
                    "content": memory.content,
                    "status": memory.status,
                    "importance_score": memory.importance_score,
                    "confidence_score": memory.confidence_score,
                    "average_relevance_score": memory.average_relevance_score,
                }
            }
        if name == "stage_status_update":
            return self.session.stage("status_update", StagedStatusUpdate.model_validate(arguments))
        if name == "stage_lineage_event":
            return self.session.stage("lineage_event", StagedLineageEvent.model_validate(arguments))
        if name == "stage_contradiction_update":
            return self.session.stage("contradiction_update", StagedContradictionUpdate.model_validate(arguments))
        if name == "stage_snapshot_selection_override":
            return self.session.stage("snapshot_selection_override", SnapshotCandidateSet.model_validate(arguments))
        if name == "finalize_cortex_review":
            return self.session.finalize(str(arguments["reasoning_summary"]))
        raise ValueError("Unknown cortex tool: {0}".format(name))

    def validate(self) -> list[str]:
        """Validate staged cortex review operations."""
        existing_memory_ids = {memory.id for memory in self.memory_repo.list_user_global_memories(self.user_id, limit=500)}
        return self.validator.validate_cortex_review(self.session, set(self.proposal_index.keys()), existing_memory_ids)


class CortexLLM:
    """Role-specific interface for hourly maintenance review and snapshot synthesis."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.review_prompt = load_prompt("cortex_review")
        self.summary_prompt = load_prompt("cortex_summary")

    def render_review_prompt(self, context: PromptRenderContext, bundle: MaintenanceProposalBundle) -> str:
        """Render the cortex maintenance review prompt."""
        return (
            "## Maintenance subject\n{0}\n\n"
            "## Active memories\n{1}\n\n"
            "## Graph facts\n{2}\n\n"
            "## Contradiction context\n{3}\n\n"
            "## Lineage context\n{4}\n\n"
            "## Programmatic proposals\n{5}\n\n"
            "## Snapshot candidates\n{6}\n\n"
            "## Extra metadata\n{7}\n\n"
            "## Tool-loop reminder\n"
            "- Inspect proposals before staging adjustments.\n"
            "- Stage only the items you approve or adjust.\n"
            "- Skip proposals by not staging them.\n"
            "- Finalize when review is complete."
        ).format(
            context.subject,
            format_memories(context.memories, limit=30),
            format_graph_facts(context.graph_facts, limit=30),
            format_string_list(context.contradiction_summaries, limit=20),
            format_string_list(context.lineage_summaries, limit=20),
            json.dumps(bundle.model_dump(mode="json"), indent=2),
            json.dumps(bundle.snapshot_candidates),
            json.dumps(context.extras, sort_keys=True),
        )

    def review_proposals(self, context: PromptRenderContext, bundle: MaintenanceProposalBundle, executor: CortexReviewToolExecutor, max_steps: int = 12) -> MaintenanceReviewResult:
        """Run the Cortex proposal-review tool loop."""
        result = self.client.run_tool_loop(
            system_prompt=self.review_prompt,
            user_prompt=self.render_review_prompt(context, bundle),
            tools=executor.tools(),
            tool_executor=executor.execute,
            max_steps=max_steps,
        )
        reasoning_summary = executor.session.reasoning_summary or result.final_content
        return MaintenanceReviewResult(
            reasoning_summary=reasoning_summary,
            tool_session=executor.session,
            tool_call_count=result.tool_call_count,
        )

    def summarize_snapshot(self, context: PromptRenderContext) -> CortexSnapshotSummary:
        """Generate the final snapshot summary from reviewed snapshot candidates."""
        payload = self.client.generate_json(
            system_prompt=self.summary_prompt,
            user_prompt=(
                "## Snapshot subject\n{0}\n\n## Selected memories\n{1}\n\n## Graph facts\n{2}\n\n## Extra metadata\n{3}"
            ).format(
                context.subject,
                format_memories(context.memories, limit=20),
                format_graph_facts(context.graph_facts, limit=20),
                json.dumps(context.extras, sort_keys=True),
            ),
            schema_name="cortex_summary",
        )
        return CortexSnapshotSummary.model_validate(payload)
