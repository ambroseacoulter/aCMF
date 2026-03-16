"""Shared LLM client abstractions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import EmbeddingRoleConfig, OpenAICompatibleRoleConfig
from app.core.enums import ProviderKind


class LLMClient(Protocol):
    """Common LLM client protocol."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> dict[str, Any]:
        """Generate a JSON object response."""

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response."""

    def run_tool_loop(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list["ToolDefinition"],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_steps: int,
    ) -> "ToolLoopResult":
        """Run an LLM tool loop and return the final response."""


class EmbeddingClient(Protocol):
    """Common embedding protocol."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts."""


class ToolDefinition(BaseModel):
    """Definition of a callable tool exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolLoopResult(BaseModel):
    """Final result from a tool loop."""

    final_content: str
    tool_call_count: int = 0


class StubLLMClient:
    """Deterministic local LLM stub."""

    def __init__(self, role_name: str) -> None:
        self.role_name = role_name

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> dict[str, Any]:
        """Return a schema-shaped stub response."""
        if schema_name == "adjudicator":
            return {
                "memory_decisions": [],
                "entity_proposals": [],
                "relation_proposals": [],
                "reasoning_summary": "No durable memory extracted from stub provider.",
            }
        if schema_name == "context_enhancer":
            return {
                "has_usable_context": bool(user_prompt.strip()),
                "summary": "Retrieved grounded memory context." if user_prompt.strip() else "",
                "active_context": user_prompt[:300],
                "confidence_note": "Use as supportive context, not unquestionable fact.",
                "used_memory_ids": [],
                "abstained_reason": None if user_prompt.strip() else "No relevant grounded memory found.",
            }
        if schema_name == "deep_memory":
            return {
                "answer": "I do not have enough grounded memory evidence to answer confidently.",
                "used_memory_ids": [],
                "abstained": True,
                "abstained_reason": "Stub provider uses conservative abstention by default.",
            }
        if schema_name in {"cortex", "cortex_summary"}:
            return {
                "summary": "Hourly snapshot generated from durable user and global memories.",
                "health_note": "Stub provider performed conservative maintenance.",
            }
        return {}

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a deterministic text response."""
        return user_prompt[:500]

    def run_tool_loop(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_steps: int,
    ) -> ToolLoopResult:
        """Run a deterministic pseudo tool loop for local development."""
        tool_names = {tool.name for tool in tools}
        tool_call_count = 0
        lowered = user_prompt.lower()
        if "search_memory" in tool_names:
            tool_executor("search_memory", {"query": user_prompt[:120], "limit": 5})
            tool_call_count += 1
        if "stage_create_memory" in tool_names and ("prefer" in lowered or "my name is" in lowered or "i am " in lowered):
            memory_type = "preference" if "prefer" in lowered else "identity"
            tool_executor(
                "stage_create_memory",
                {
                    "memory_type": memory_type,
                    "content": user_prompt[:280],
                    "summary": user_prompt[:120],
                    "rationale": "Stub provider detected a durable candidate from the turn.",
                    "evidence": [user_prompt[:160]],
                    "importance_score": 0.65,
                    "confidence_score": 0.72,
                    "novelty_score": 0.60,
                    "initial_relevance_score": 0.66,
                    "contradiction_risk": 0.10,
                    "target_scopes": ["user"],
                },
            )
            tool_call_count += 1
        if "finalize_adjudication" in tool_names:
            tool_executor("finalize_adjudication", {"reasoning_summary": "Stub adjudicator completed a bounded staged workflow."})
            tool_call_count += 1
            return ToolLoopResult(final_content=json.dumps({"reasoning_summary": "Stub adjudicator completed a bounded staged workflow."}), tool_call_count=tool_call_count)
        if "finalize_cortex_review" in tool_names:
            tool_executor("finalize_cortex_review", {"reasoning_summary": "Stub cortex reviewed the generated proposals."})
            tool_call_count += 1
            return ToolLoopResult(final_content=json.dumps({"reasoning_summary": "Stub cortex reviewed the generated proposals."}), tool_call_count=tool_call_count)
        return ToolLoopResult(final_content=json.dumps({"reasoning_summary": "Stub tool loop completed."}), tool_call_count=tool_call_count)


class OpenAICompatibleClient:
    """OpenAI-compatible chat client with per-role endpoint and key."""

    def __init__(self, config: OpenAICompatibleRoleConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> dict[str, Any]:
        """Ask the model for structured JSON, with a text fallback parser."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return self._parse_json(content)

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Ask the model for a text response."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def run_tool_loop(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[ToolDefinition],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_steps: int,
    ) -> ToolLoopResult:
        """Run a bounded function-calling loop using OpenAI-compatible tools."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        tool_call_count = 0
        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return ToolLoopResult(final_content=message.content or "{}", tool_call_count=tool_call_count)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)
            for call in tool_calls:
                arguments = json.loads(call.function.arguments or "{}")
                result = tool_executor(call.function.name, arguments)
                tool_call_count += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
        raise RuntimeError("LLM tool loop exceeded max_steps")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON content, handling fenced or prefixed responses."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            stripped = content.strip()
            if "```json" in stripped:
                stripped = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in stripped:
                stripped = stripped.split("```", 1)[1].split("```", 1)[0].strip()
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end >= start:
                return json.loads(stripped[start : end + 1])
            raise


class StubEmbeddingClient:
    """Deterministic local embedding provider."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate simple deterministic embeddings."""
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(ord(char) for char in text) % 997
            vectors.append([((seed + idx * 31) % 257) / 257.0 for idx in range(self.dimensions)])
        return vectors


class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible embeddings client."""

    def __init__(self, config: EmbeddingRoleConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings API."""
        response = self.client.embeddings.create(
            model=self.config.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


def resolve_llm_client(role_name: str, config: OpenAICompatibleRoleConfig) -> LLMClient:
    """Resolve a chat client for a role."""
    if config.provider == ProviderKind.STUB:
        return StubLLMClient(role_name)
    return OpenAICompatibleClient(config)


def resolve_embedding_client(config: EmbeddingRoleConfig) -> EmbeddingClient:
    """Resolve an embedding client."""
    if config.provider == ProviderKind.STUB:
        return StubEmbeddingClient(config.dimensions)
    return OpenAICompatibleEmbeddingClient(config)


def load_prompt(name: str) -> str:
    """Load a prompt template file by name."""
    path = Path(__file__).parent / "prompts" / "{0}.txt".format(name)
    text = path.read_text(encoding="utf-8")
    shared_dir = Path(__file__).parent / "prompts"
    text = text.replace("{{shared_memory_policy}}", (shared_dir / "shared_memory_policy.txt").read_text(encoding="utf-8"))
    text = text.replace("{{shared_graph_policy}}", (shared_dir / "shared_graph_policy.txt").read_text(encoding="utf-8"))
    return text
