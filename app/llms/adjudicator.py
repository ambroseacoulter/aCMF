"""Adjudicator role integration."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel

from app.llms.client import LLMClient, ToolDefinition, load_prompt
from app.llms.prompting import PromptRenderContext, format_graph_facts, format_memories, format_string_list
from app.services.tool_session_service import (
    StagedContradictionUpdate,
    StagedGraphEntity,
    StagedGraphRelation,
    StagedMemoryCreate,
    StagedMemoryEntityLink,
    StagedMemoryMerge,
    StagedMemoryUpdate,
    ToolSession,
    ToolSessionValidator,
)


class AdjudicationCandidate(BaseModel):
    """Extracted durable-memory candidate for adjudication."""

    candidate_id: str
    claim: str
    candidate_type: str
    reason: str


@dataclass
class AdjudicationRunResult:
    """Final result from an adjudicator tool run."""

    reasoning_summary: str
    tool_session: ToolSession
    tool_call_count: int


class AdjudicationToolExecutor:
    """Read tools and staged write tools available to the adjudicator."""

    def __init__(
        self,
        *,
        user_id: str,
        container_ids: list[str],
        memory_repo,
        vector_search,
        graph_engine,
    ) -> None:
        self.user_id = user_id
        self.container_ids = container_ids
        self.memory_repo = memory_repo
        self.vector_search = vector_search
        self.graph_engine = graph_engine
        self.session = ToolSession(session_id="adj_{0}".format(user_id), kind="adjudication", user_id=user_id)
        self.validator = ToolSessionValidator()

    def tools(self) -> list[ToolDefinition]:
        """Return available adjudicator tools."""
        return [
            ToolDefinition(
                name="search_memory",
                description="Semantic search across the allowed scoped memory corpus.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="search_memory_metadata",
                description="Metadata search by status and memory type across the allowed scoped corpus.",
                parameters={
                    "type": "object",
                    "properties": {
                        "statuses": {"type": "array", "items": {"type": "string"}},
                        "memory_types": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "default": 5},
                    },
                },
            ),
            ToolDefinition(
                name="lookup_contradiction_groups",
                description="Inspect open contradiction groups for this user.",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                },
            ),
            ToolDefinition(
                name="lookup_lineage",
                description="Inspect recent lineage events or events for a specific memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                },
            ),
            ToolDefinition(
                name="lookup_graph",
                description="Inspect graph neighborhood facts relevant to a query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="fetch_memory",
                description="Fetch a specific memory record by id.",
                parameters={
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            ),
            ToolDefinition(
                name="stage_create_memory",
                description="Stage creation of a new memory at one or more scopes.",
                parameters=StagedMemoryCreate.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_update_memory",
                description="Stage an update to an existing memory.",
                parameters=StagedMemoryUpdate.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_merge_memory",
                description="Stage a merge of a new candidate into an existing memory.",
                parameters=StagedMemoryMerge.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_mark_contradiction",
                description="Stage a contradiction update between an existing memory and a new or existing counter-memory.",
                parameters=StagedContradictionUpdate.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_create_entity",
                description="Stage a graph entity creation/update.",
                parameters=StagedGraphEntity.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_create_relation",
                description="Stage a graph relation creation.",
                parameters=StagedGraphRelation.model_json_schema(),
            ),
            ToolDefinition(
                name="stage_link_memory_entity",
                description="Stage a link between a memory reference and an entity name.",
                parameters=StagedMemoryEntityLink.model_json_schema(),
            ),
            ToolDefinition(
                name="finalize_adjudication",
                description="Finalize adjudication once all investigations and staged operations are complete.",
                parameters={
                    "type": "object",
                    "properties": {"reasoning_summary": {"type": "string"}},
                    "required": ["reasoning_summary"],
                },
            ),
        ]

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Execute one adjudicator tool call."""
        if name == "search_memory":
            limit = int(arguments.get("limit", 5))
            memories = self.vector_search.search(
                user_id=self.user_id,
                scope_types=["user", "global", "container"],
                bucket_ids=self.container_ids,
                query=str(arguments["query"]),
                limit=limit,
            )
            return {"results": [self._memory_result(memory) for memory in memories]}
        if name == "search_memory_metadata":
            memories = self.memory_repo.search_by_metadata(
                self.user_id,
                ["user", "global", "container"],
                self.container_ids,
                statuses=list(arguments.get("statuses", []) or []),
                memory_types=list(arguments.get("memory_types", []) or []),
                limit=int(arguments.get("limit", 5)),
            )
            return {"results": [self._memory_result(memory) for memory in memories]}
        if name == "lookup_contradiction_groups":
            groups = self.memory_repo.list_open_contradiction_groups(
                self.user_id,
                topic=str(arguments.get("topic")) if arguments.get("topic") else None,
                limit=int(arguments.get("limit", 5)),
            )
            return {
                "results": [
                    {
                        "group_id": group.id,
                        "topic": group.topic,
                        "description": group.description,
                        "status": group.status,
                    }
                    for group in groups
                ]
            }
        if name == "lookup_lineage":
            events = self.memory_repo.list_lineage_events(
                self.user_id,
                memory_id=str(arguments.get("memory_id")) if arguments.get("memory_id") else None,
                limit=int(arguments.get("limit", 5)),
            )
            return {
                "results": [
                    {
                        "source_memory_id": event.source_memory_id,
                        "target_memory_id": event.target_memory_id,
                        "event_type": event.event_type,
                        "confidence_score": event.confidence_score,
                    }
                    for event in events
                ]
            }
        if name == "lookup_graph":
            graph_result = self.graph_engine.traverse(self.user_id, str(arguments["query"]), int(arguments.get("limit", 5)))
            return {"results": [fact.__dict__ for fact in graph_result.facts], "graph_density_signal": graph_result.graph_density_signal}
        if name == "fetch_memory":
            memory = self.memory_repo.get(str(arguments["memory_id"]))
            return {"result": self._memory_result(memory) if memory else None}
        if name == "stage_create_memory":
            payload = StagedMemoryCreate.model_validate(arguments)
            return self.session.stage("create_memory", payload)
        if name == "stage_update_memory":
            payload = StagedMemoryUpdate.model_validate(arguments)
            return self.session.stage("update_memory", payload)
        if name == "stage_merge_memory":
            payload = StagedMemoryMerge.model_validate(arguments)
            return self.session.stage("merge_memory", payload)
        if name == "stage_mark_contradiction":
            payload = StagedContradictionUpdate.model_validate(arguments)
            return self.session.stage("mark_contradiction", payload)
        if name == "stage_create_entity":
            payload = StagedGraphEntity.model_validate(arguments)
            return self.session.stage("create_entity", payload)
        if name == "stage_create_relation":
            payload = StagedGraphRelation.model_validate(arguments)
            return self.session.stage("create_relation", payload)
        if name == "stage_link_memory_entity":
            payload = StagedMemoryEntityLink.model_validate(arguments)
            return self.session.stage("link_memory_entity", payload)
        if name == "finalize_adjudication":
            return self.session.finalize(str(arguments["reasoning_summary"]))
        raise ValueError("Unknown adjudicator tool: {0}".format(name))

    def validate(self, allowed_scopes: list[str] | None = None) -> list[str]:
        """Validate the staged adjudication operations."""
        existing_memory_ids = {memory.id for memory in self.memory_repo.list_recent_candidates(self.user_id, ["user", "global", "container"], self.container_ids, limit=500)}
        return self.validator.validate_adjudication(self.session, allowed_scopes or ["user", "global", "container"], existing_memory_ids)

    @staticmethod
    def _memory_result(memory) -> dict[str, object]:
        """Serialize a memory for tool output."""
        return {
            "memory_id": memory.id,
            "scope_type": memory.scope_type,
            "bucket_id": memory.bucket_id,
            "memory_type": memory.memory_type,
            "status": memory.status,
            "summary": memory.summary,
            "content": memory.content,
            "importance_score": memory.importance_score,
            "confidence_score": memory.confidence_score,
            "current_relevance_score": memory.current_relevance_score,
            "contradiction_risk": memory.contradiction_risk,
        }


class AdjudicatorLLM:
    """Role-specific interface for post-turn adjudication."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.system_prompt = load_prompt("adjudicator")

    def render_user_prompt(self, context: PromptRenderContext) -> str:
        """Render the user prompt payload."""
        return (
            "## Turn subject\n{0}\n\n"
            "## Retrieved candidate memories\n{1}\n\n"
            "## Graph facts\n{2}\n\n"
            "## Active contradiction context\n{3}\n\n"
            "## Existing lineage context\n{4}\n\n"
            "## Tool-loop budgets\n{5}\n\n"
            "## Extra metadata\n{6}\n\n"
            "## Tool-loop reminder\n"
            "- Search before staging writes.\n"
            "- Stage one operation at a time.\n"
            "- Finalize only when the staged set is complete.\n"
            "- Do not assume facts you have not inspected."
        ).format(
            context.subject,
            format_memories(context.memories),
            format_graph_facts(context.graph_facts),
            format_string_list(context.contradiction_summaries),
            format_string_list(context.lineage_summaries),
            json.dumps(context.budgets, sort_keys=True),
            json.dumps(context.extras, sort_keys=True),
        )

    def run_with_tools(self, context: PromptRenderContext, executor: AdjudicationToolExecutor, max_steps: int = 12) -> AdjudicationRunResult:
        """Run the adjudicator tool loop."""
        result = self.client.run_tool_loop(
            system_prompt=self.system_prompt,
            user_prompt=self.render_user_prompt(context),
            tools=executor.tools(),
            tool_executor=executor.execute,
            max_steps=max_steps,
        )
        reasoning_summary = executor.session.reasoning_summary or result.final_content
        return AdjudicationRunResult(
            reasoning_summary=reasoning_summary,
            tool_session=executor.session,
            tool_call_count=result.tool_call_count,
        )
