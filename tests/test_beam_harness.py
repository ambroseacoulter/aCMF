"""Tests for the BEAM benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from app.benchmarks.beam.client import JobStatusHistory
from app.benchmarks.beam.config import BeamHarnessConfig
from app.benchmarks.beam.dataset import discover_chat_fixtures
from app.benchmarks.beam.evaluator import build_evaluation_summary
from app.benchmarks.beam.runner import run_tier


def test_discover_chat_fixtures_extracts_turns_and_questions(tmp_path: Path) -> None:
    beam_root = _write_beam_fixture(tmp_path)
    config = BeamHarnessConfig()

    fixtures = discover_chat_fixtures(beam_root, "100K", config)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.chat_index == "0"
    assert len(fixture.turns) == 2
    assert fixture.turns[0].user_message == "User says hello"
    assert fixture.turns[1].assistant_response == "Assistant shares second reply"
    assert fixture.probing_questions["information_extraction"][0].question == "What greeting did the user give?"


def test_run_tier_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    beam_root = _write_beam_fixture(tmp_path)
    output_directory = tmp_path / "artifacts"
    config = BeamHarnessConfig()

    monkeypatch.setattr("app.benchmarks.beam.client.BeamAPIClient.healthcheck", lambda self: {"status": "ok"})
    monkeypatch.setattr(
        "app.benchmarks.beam.client.BeamAPIClient.process_turn",
        lambda self, payload: {
            "status": "accepted",
            "job_id": "job-{0}".format(payload["turn"]["user_message"][-1]),
            "created_user": False,
            "created_containers": [],
            "accepted_at": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        "app.benchmarks.beam.client.BeamAPIClient.wait_for_job",
        lambda self, job_id: JobStatusHistory(
            job_id=job_id,
            statuses=[
                {"job_id": job_id, "status": "running"},
                {"job_id": job_id, "status": "succeeded"},
            ],
        ),
    )
    monkeypatch.setattr(
        "app.benchmarks.beam.client.BeamAPIClient.answer_deep_memory",
        lambda self, payload: {
            "status": "ok",
            "answer": "answer::{0}".format(payload["query"]),
            "abstained": False,
            "abstained_reason": None,
            "used_memory_count": 1,
            "diagnostics": {"candidate_count": 1},
            "evidence": [{"memory_id": "mem-1"}],
        },
    )

    result = run_tier(
        beam_root=beam_root,
        output_directory=output_directory,
        tier="100K",
        run_id="run-1",
        config=config,
        api_base_url="http://localhost:8000",
        image_ref="acmf:test",
    )

    answer_file = result.results_directory / "0" / config.export.result_filename
    exported = json.loads(answer_file.read_text())
    assert exported["information_extraction"][0]["llm_response"] == "answer::What greeting did the user give?"
    assert exported["abstention"][0]["question_type"] == "abstention"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["docker_image_ref"] == "acmf:test"
    assert manifest["tier"] == "100K"
    failure_report = json.loads(result.failure_report_path.read_text())
    assert failure_report["failures"] == []


def test_build_evaluation_summary_aggregates_numeric_scores(tmp_path: Path) -> None:
    config = BeamHarnessConfig()
    results_directory = tmp_path / "results" / "100K" / "0"
    results_directory.mkdir(parents=True)
    evaluation_path = results_directory / "{0}{1}".format(config.export.evaluation_prefix, config.export.result_filename)
    evaluation_path.write_text(
        json.dumps(
            {
                "information_extraction": [{"score": 1.0}, {"score": 0.0}],
                "abstention": [{"result": 0.5}],
            }
        )
    )

    summary = build_evaluation_summary(results_directory.parent, config)

    assert summary["overall"]["information_extraction"]["average_score"] == 0.5
    assert summary["overall"]["abstention"]["average_score"] == 0.5


def _write_beam_fixture(tmp_path: Path) -> Path:
    beam_root = tmp_path / "BEAM"
    chat_dir = beam_root / "chats" / "100K" / "0"
    probing_dir = chat_dir / "probing_questions"
    probing_dir.mkdir(parents=True)
    (chat_dir / "conversation.json").write_text(
        json.dumps(
            [
                {
                    "turns": [
                        [
                            {"role": "user", "content": "User says hello"},
                            {"role": "assistant", "content": "Assistant replies hello"},
                        ],
                        [
                            {"role": "user", "content": "User asks a follow-up"},
                            {"role": "assistant", "content": "Assistant shares second reply"},
                        ],
                    ]
                }
            ]
        )
    )
    (probing_dir / "probing_questions.json").write_text(
        json.dumps(
            {
                "information_extraction": [
                    {"question": "What greeting did the user give?", "rubric": "The answer should mention hello."}
                ],
                "abstention": [
                    {"question": "What was the user's middle name?", "rubric": "The model should abstain."}
                ],
            }
        )
    )
    return beam_root
