"""Context enhancer role integration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.core.enums import ScopeLevel
from app.llms.client import LLMClient, load_prompt
from app.llms.prompting import PromptRenderContext, format_graph_facts, format_memories, format_string_list


class ContextEnhancementPayload(BaseModel):
    """Structured context enhancement payload."""

    has_usable_context: bool = False
    summary: str = ""
    active_context: str = ""
    confidence_note: str = "Use as supportive context, not unquestionable fact."
    used_memory_ids: list[str] = Field(default_factory=list)
    abstained_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_null_strings(cls, value: object) -> object:
        """Treat provider-emitted null string fields as empty defaults."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name, default in {
            "summary": "",
            "active_context": "",
            "confidence_note": "Use as supportive context, not unquestionable fact.",
        }.items():
            if normalized.get(field_name) is None:
                normalized[field_name] = default
        return normalized


class DeepMemoryPayload(BaseModel):
    """Structured deep-memory answer payload."""

    answer: str
    used_memory_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstained_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_null_answer(cls, value: object) -> object:
        """Treat provider-emitted null answers as an empty answer."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if normalized.get("answer") is None:
            normalized["answer"] = ""
        return normalized


class ContextEnhancerLLM:
    """Role-specific synthesis interface."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.context_prompt = load_prompt("context_enhancer")
        self.deep_memory_prompt = load_prompt("deep_memory")

    def _render_shared_block(self, context: PromptRenderContext) -> str:
        """Render shared context for context/deep-memory prompts."""
        return (
            "## Subject\n{0}\n\n"
            "## Retrieved memories\n{1}\n\n"
            "## Graph facts\n{2}\n\n"
            "## Contradiction context\n{3}\n\n"
            "## Lineage context\n{4}\n\n"
            "## Budgets\n{5}\n\n"
            "## Extra metadata\n{6}"
        ).format(
            context.subject,
            format_memories(context.memories),
            format_graph_facts(context.graph_facts),
            format_string_list(context.contradiction_summaries),
            format_string_list(context.lineage_summaries),
            context.budgets,
            context.extras,
        )

    def synthesize_context(self, scope_level: ScopeLevel, context: PromptRenderContext) -> ContextEnhancementPayload:
        """Synthesize prompt-ready context."""
        payload = self.client.generate_json(
            system_prompt=self.context_prompt,
            user_prompt="scope_level={0}\n\n{1}".format(scope_level, self._render_shared_block(context)),
            schema_name="context_enhancer",
        )
        return ContextEnhancementPayload.model_validate(payload)

    def answer_deep_memory(self, context: PromptRenderContext) -> DeepMemoryPayload:
        """Answer a deep-memory query using retrieved evidence only."""
        payload = self.client.generate_json(
            system_prompt=self.deep_memory_prompt,
            user_prompt=self._render_shared_block(context),
            schema_name="deep_memory",
        )
        return DeepMemoryPayload.model_validate(payload)


def render_context_xml(scope_level: ScopeLevel, payload: ContextEnhancementPayload) -> str:
    """Render the stable XML-like context block."""
    return (
        "<contextenhancement>\n"
        "  <scope>{0}</scope>\n"
        "  <summary>{1}</summary>\n"
        "  <active_context>{2}</active_context>\n"
        "  <confidence_note>{3}</confidence_note>\n"
        "</contextenhancement>"
    ).format(scope_level, payload.summary, payload.active_context, payload.confidence_note)
