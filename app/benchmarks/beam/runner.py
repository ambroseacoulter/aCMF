"""Benchmark orchestration for BEAM against aCMF."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.benchmarks.beam.client import BeamAPIClient
from app.benchmarks.beam.config import BeamHarnessConfig
from app.benchmarks.beam.dataset import BeamChatFixture, BeamQuestion, discover_chat_fixtures

logger = logging.getLogger(__name__)


@dataclass
class BeamRunResult:
    """Paths produced by one benchmark run."""

    run_directory: Path
    results_directory: Path
    manifest_path: Path
    failure_report_path: Path


def run_tier(
    *,
    beam_root: Path,
    output_directory: Path,
    tier: str,
    run_id: str,
    config: BeamHarnessConfig,
    api_base_url: str | None = None,
    image_ref: str | None = None,
) -> BeamRunResult:
    """Run the aCMF BEAM harness for one tier."""
    runtime_config = config.model_copy(deep=True)
    if api_base_url:
        runtime_config.runtime.api_base_url = api_base_url
    if image_ref:
        runtime_config.docker.image_ref = image_ref
    fixtures = discover_chat_fixtures(beam_root, tier, runtime_config)
    canonical_tier = config.canonical_tier(tier)
    run_directory = output_directory / run_id
    results_directory = run_directory / "results" / canonical_tier
    results_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / "manifests" / "{0}.json".format(canonical_tier)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failure_report_path = run_directory / "failures" / "{0}.json".format(canonical_tier)
    failure_report_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    start_time = _utc_now()
    logger.info(
        "Starting BEAM tier run tier=%s run_id=%s chats=%s api_base_url=%s image_ref=%s",
        canonical_tier,
        run_id,
        len(fixtures),
        runtime_config.runtime.api_base_url,
        runtime_config.docker.image_ref,
    )
    client = BeamAPIClient(runtime_config, base_url=runtime_config.runtime.api_base_url)
    try:
        client.healthcheck()
        logger.info("aCMF API healthcheck succeeded")
        for fixture in fixtures:
            try:
                logger.info(
                    "Running chat chat_index=%s turns=%s question_categories=%s total_questions=%s",
                    fixture.chat_index,
                    len(fixture.turns),
                    len(fixture.probing_questions),
                    sum(len(items) for items in fixture.probing_questions.values()),
                )
                _run_chat_fixture(
                    client=client,
                    fixture=fixture,
                    run_id=run_id,
                    results_directory=results_directory,
                    config=runtime_config,
                )
            except Exception as exc:
                logger.exception("Chat failed chat_index=%s error=%s", fixture.chat_index, exc)
                failures.append(
                    {
                        "chat_index": fixture.chat_index,
                        "error": str(exc),
                        "conversation_file": str(fixture.conversation_file),
                        "probing_questions_file": str(fixture.probing_questions_file),
                    }
                )
                break
    finally:
        client.close()

    end_time = _utc_now()
    manifest = build_manifest(
        beam_root=beam_root,
        config=runtime_config,
        run_id=run_id,
        tier=canonical_tier,
        image_ref=runtime_config.docker.image_ref,
        start_time=start_time,
        end_time=end_time,
        results_directory=results_directory,
        failure_count=len(failures),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    failure_report_path.write_text(json.dumps({"failures": failures}, indent=2, sort_keys=True))
    logger.info(
        "Completed BEAM tier run tier=%s run_id=%s failures=%s manifest=%s",
        canonical_tier,
        run_id,
        len(failures),
        manifest_path,
    )
    return BeamRunResult(
        run_directory=run_directory,
        results_directory=results_directory,
        manifest_path=manifest_path,
        failure_report_path=failure_report_path,
    )


def build_manifest(
    *,
    beam_root: Path,
    config: BeamHarnessConfig,
    run_id: str,
    tier: str,
    image_ref: str,
    start_time: datetime,
    end_time: datetime,
    results_directory: Path,
    failure_count: int,
) -> dict[str, Any]:
    """Return one manifest payload."""
    return {
        "run_id": run_id,
        "tier": tier,
        "beam_commit": _git_commit(beam_root) or config.dataset.repo_commit,
        "acmf_commit": _git_commit(_repo_root()),
        "docker_image_ref": image_ref,
        "provider_metadata": config.provider_metadata or _provider_metadata_from_env(),
        "evaluator_metadata": config.evaluator_metadata,
        "harness_config": config.model_dump(mode="json"),
        "results_directory": str(results_directory),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "failure_count": failure_count,
    }


def _run_chat_fixture(
    *,
    client: BeamAPIClient,
    fixture: BeamChatFixture,
    run_id: str,
    results_directory: Path,
    config: BeamHarnessConfig,
) -> None:
    """Ingest one chat and answer its probing questions."""
    chat_output_dir = results_directory / fixture.chat_index
    chat_output_dir.mkdir(parents=True, exist_ok=True)
    user_id = "beam-{0}-{1}-{2}".format(run_id, fixture.tier.lower(), fixture.chat_index)
    process_acks: list[dict[str, Any]] = []
    job_histories: list[dict[str, Any]] = []
    response_artifacts: list[dict[str, Any]] = []
    exported_answers: dict[str, list[dict[str, Any]]] = {}
    logger.info("Chat start chat_index=%s user_id=%s", fixture.chat_index, user_id)

    for turn in fixture.turns:
        process_payload = {
            "user_id": user_id,
            "containers": [],
            "scope_policy": {"write_user": "true", "write_global": "false", "write_container": False},
            "turn": {
                "user_message": turn.user_message,
                "assistant_response": turn.assistant_response,
                "occurred_at": _synthetic_occurred_at(turn.index).isoformat(),
            },
            "metadata": {
                "app": "beam-harness",
                "source": "beam",
                "trace_id": "{0}:{1}:{2}:turn:{3}".format(run_id, fixture.tier, fixture.chat_index, turn.index),
            },
        }
        ack = client.process_turn(process_payload)
        process_acks.append(ack)
        logger.info(
            "Turn accepted chat_index=%s turn=%s/%s job_id=%s",
            fixture.chat_index,
            turn.index + 1,
            len(fixture.turns),
            ack["job_id"],
        )
        history = client.wait_for_job(str(ack["job_id"]))
        job_histories.append({"job_id": history.job_id, "statuses": history.statuses})
        final_status = history.statuses[-1]["status"]
        logger.info(
            "Turn completed chat_index=%s turn=%s/%s job_id=%s status=%s polls=%s",
            fixture.chat_index,
            turn.index + 1,
            len(fixture.turns),
            history.job_id,
            final_status,
            len(history.statuses),
        )
        if final_status != "succeeded":
            raise RuntimeError("Job {0} failed for chat {1}".format(history.job_id, fixture.chat_index))

    for category, questions in fixture.probing_questions.items():
        exported_answers.setdefault(category, [])
        logger.info(
            "Answering category chat_index=%s category=%s questions=%s",
            fixture.chat_index,
            category,
            len(questions),
        )
        for question in questions:
            deep_memory_payload = {
                "user_id": user_id,
                "query": question.question,
                "containers": [],
                "scope_level": "user",
                "read_mode": "deep",
                "metadata": {
                    "app": "beam-harness",
                    "source": "beam",
                    "trace_id": "{0}:{1}:{2}:question:{3}:{4}".format(
                        run_id,
                        fixture.tier,
                        fixture.chat_index,
                        category,
                        question.index,
                    ),
                    "model": "acmf-deep-memory",
                },
            }
            response = client.answer_deep_memory(deep_memory_payload)
            response_artifacts.append(
                {
                    "category": category,
                    "question_index": question.index,
                    "question": question.question,
                    "response": response,
                }
            )
            exported_answers[category].append(_export_answer(question, response))
            logger.info(
                "Question answered chat_index=%s category=%s question=%s/%s abstained=%s used_memory_count=%s",
                fixture.chat_index,
                category,
                question.index + 1,
                len(questions),
                response.get("abstained"),
                response.get("used_memory_count"),
            )

    (chat_output_dir / "ingestion-log.json").write_text(
        json.dumps(
            {
                "chat_index": fixture.chat_index,
                "conversation_file": str(fixture.conversation_file),
                "probing_questions_file": str(fixture.probing_questions_file),
                "turn_count": len(fixture.turns),
            },
            indent=2,
            sort_keys=True,
        )
    )
    (chat_output_dir / "process-acks.json").write_text(json.dumps(process_acks, indent=2, sort_keys=True))
    (chat_output_dir / "job-status-history.json").write_text(json.dumps(job_histories, indent=2, sort_keys=True))
    (chat_output_dir / "deep-memory-responses.json").write_text(json.dumps(response_artifacts, indent=2, sort_keys=True))
    (chat_output_dir / config.export.result_filename).write_text(json.dumps(exported_answers, indent=2, sort_keys=True))
    logger.info("Chat completed chat_index=%s output_dir=%s", fixture.chat_index, chat_output_dir)


def _export_answer(question: BeamQuestion, response: dict[str, Any]) -> dict[str, Any]:
    """Build one BEAM answer entry."""
    return {
        "question": question.question,
        "llm_response": response["answer"],
        "abstained": response.get("abstained"),
        "abstained_reason": response.get("abstained_reason"),
        "used_memory_count": response.get("used_memory_count"),
        "diagnostics": response.get("diagnostics"),
        "evidence": response.get("evidence"),
        "question_index": question.index,
        "question_type": question.category,
    }


def _provider_metadata_from_env() -> dict[str, str]:
    """Capture a lightweight provider/model snapshot from the environment."""
    keys = [
        "ACMF_ADJUDICATOR_PROVIDER",
        "ACMF_ADJUDICATOR_MODEL",
        "ACMF_CONTEXT_ENHANCER_PROVIDER",
        "ACMF_CONTEXT_ENHANCER_MODEL",
        "ACMF_CORTEX_PROVIDER",
        "ACMF_CORTEX_MODEL",
        "ACMF_EMBEDDING_PROVIDER",
        "ACMF_EMBEDDING_MODEL",
    ]
    return {key: value for key, value in ((key, os.environ.get(key)) for key in keys) if value}


def _synthetic_occurred_at(index: int) -> datetime:
    """Return a deterministic synthetic occurred-at timestamp."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(seconds=index)


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _git_commit(path: Path) -> str | None:
    """Return the current git commit for a repository, if available."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _repo_root() -> Path:
    """Return the local aCMF repository root."""
    return Path(__file__).resolve().parents[3]
