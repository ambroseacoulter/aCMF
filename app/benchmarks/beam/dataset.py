"""Dataset discovery and adaptation helpers for BEAM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.benchmarks.beam.config import BeamHarnessConfig


@dataclass(frozen=True)
class BeamTurn:
    """One turn pair adapted for `/v1/process`."""

    index: int
    user_message: str
    assistant_response: str


@dataclass(frozen=True)
class BeamQuestion:
    """One probing question from the benchmark."""

    category: str
    index: int
    question: str
    raw_question: dict[str, Any]


@dataclass(frozen=True)
class BeamChatFixture:
    """Resolved dataset inputs for one BEAM chat."""

    tier: str
    chat_index: str
    chat_dir: Path
    conversation_file: Path
    probing_questions_file: Path
    turns: list[BeamTurn]
    probing_questions: dict[str, list[BeamQuestion]]


def discover_chat_fixtures(beam_root: Path, tier: str, config: BeamHarnessConfig) -> list[BeamChatFixture]:
    """Return all resolved chat fixtures for one tier."""
    tier_dir = beam_root / config.dataset.chats_directory / config.canonical_tier(tier)
    if not tier_dir.exists():
        raise FileNotFoundError("Missing BEAM tier directory: {0}".format(tier_dir))
    chat_dirs = sorted(
        [path for path in tier_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    fixtures: list[BeamChatFixture] = []
    for chat_dir in chat_dirs:
        probing_questions_file = chat_dir / config.dataset.probing_questions_file
        conversation_file = discover_conversation_file(chat_dir, config)
        probing_questions = load_probing_questions(probing_questions_file)
        turns = load_turns(conversation_file)
        fixtures.append(
            BeamChatFixture(
                tier=config.canonical_tier(tier),
                chat_index=chat_dir.name,
                chat_dir=chat_dir,
                conversation_file=conversation_file,
                probing_questions_file=probing_questions_file,
                turns=turns,
                probing_questions=probing_questions,
            )
        )
    return fixtures


def discover_conversation_file(chat_dir: Path, config: BeamHarnessConfig) -> Path:
    """Return the most likely conversation JSON file for one chat directory."""
    candidates = [
        path
        for path in chat_dir.rglob("*.json")
        if "probing_questions" not in path.parts and not path.name.startswith("evaluation-")
    ]
    if not candidates:
        raise FileNotFoundError("No conversation JSON file found under {0}".format(chat_dir))
    preferred_names = set(config.dataset.conversation_file_names)
    preferred = [path for path in candidates if path.name in preferred_names]
    ranked = preferred or candidates
    ranked = sorted(ranked, key=lambda path: (-path.stat().st_size, len(path.parts), path.name))
    return ranked[0]


def load_probing_questions(path: Path) -> dict[str, list[BeamQuestion]]:
    """Load probing questions grouped by question type."""
    payload = json.loads(path.read_text())
    grouped: dict[str, list[BeamQuestion]] = {}
    for category, items in payload.items():
        questions: list[BeamQuestion] = []
        for index, item in enumerate(items):
            questions.append(
                BeamQuestion(
                    category=category,
                    index=index,
                    question=str(item["question"]),
                    raw_question=dict(item),
                )
            )
        grouped[category] = questions
    return grouped


def load_turns(path: Path) -> list[BeamTurn]:
    """Load conversation content and flatten it into turn pairs."""
    payload = json.loads(path.read_text())
    turns: list[BeamTurn] = []
    for raw_turn in _iter_turns(payload):
        messages = [item for item in raw_turn if isinstance(item, dict)]
        if not messages:
            continue
        for index in range(0, len(messages), 2):
            pair = messages[index : index + 2]
            if not pair:
                continue
            user_message = _message_content(pair[0])
            assistant_response = _message_content(pair[1]) if len(pair) > 1 else ""
            turns.append(
                BeamTurn(
                    index=len(turns),
                    user_message=user_message,
                    assistant_response=assistant_response,
                )
            )
    if not turns:
        raise ValueError("No turn pairs could be extracted from {0}".format(path))
    return turns


def _iter_turns(payload: Any) -> list[list[dict[str, Any]]]:
    """Return all raw turn arrays contained in the conversation payload."""
    collected: list[list[dict[str, Any]]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            turns = node.get("turns")
            if isinstance(turns, list):
                for item in turns:
                    if isinstance(item, list):
                        collected.append(item)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return collected


def _message_content(message: dict[str, Any]) -> str:
    """Extract text content from one message object."""
    content = message.get("content")
    if isinstance(content, list):
        parts = [str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()
