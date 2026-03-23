"""Configuration models for the BEAM harness."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class BeamDatasetConfig(BaseModel):
    """Pinned dataset and file-layout configuration."""

    repo_commit: str = "PIN_ME"
    chats_directory: str = "chats"
    probing_questions_file: str = "probing_questions/probing_questions.json"
    conversation_file_names: list[str] = Field(
        default_factory=lambda: [
            "chat.json",
            "messages.json",
            "conversation.json",
            "dialogue.json",
            "dialog.json",
            "chat_messages.json",
        ]
    )
    supported_tiers: list[str] = Field(default_factory=lambda: ["100K", "500K", "1M", "10M"])
    tier_aliases: dict[str, str] = Field(default_factory=lambda: {"128K": "100K"})


class RuntimeConfig(BaseModel):
    """HTTP runtime behavior for the harness."""

    api_base_url: str = "http://localhost:8000"
    request_timeout_seconds: float = 60.0
    transport_retries: int = 3
    transport_backoff_seconds: float = 2.0
    job_poll_interval_seconds: float = 2.0
    job_timeout_seconds: float = 1800.0
    max_workers: int = 10


class ExportConfig(BaseModel):
    """Answer export naming configuration."""

    result_filename: str = "acmf-deep-memory.json"
    evaluation_prefix: str = "evaluation-"
    report_filename: str = "acmf-beam-report.xlsx"


class EvaluatorConfig(BaseModel):
    """Official BEAM evaluator paths and defaults."""

    evaluation_script: str = "src/evaluation/run_evaluation.py"
    report_script: str = "src/evaluation/report_results.py"
    model_name: str = "gpt_llm"


class DockerConfig(BaseModel):
    """Docker metadata recorded in manifests."""

    image_ref: str = "acmf:beam-benchmark"


class BeamHarnessConfig(BaseModel):
    """Top-level harness config."""

    dataset: BeamDatasetConfig = Field(default_factory=BeamDatasetConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    provider_metadata: dict = Field(default_factory=dict)
    evaluator_metadata: dict = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "BeamHarnessConfig":
        """Load config from a JSON file."""
        return cls.model_validate(json.loads(path.read_text()))

    def canonical_tier(self, tier: str) -> str:
        """Normalize a requested tier to a configured dataset tier."""
        normalized = tier.strip().upper()
        return self.dataset.tier_aliases.get(normalized, normalized)
