"""Wrapper around the official BEAM evaluator scripts."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.benchmarks.beam.config import BeamHarnessConfig

logger = logging.getLogger(__name__)


def run_beam_evaluation(
    *,
    beam_root: Path,
    results_directory: Path,
    tier: str,
    config: BeamHarnessConfig,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict[str, Any]:
    """Run the official BEAM evaluator against one tier of results."""
    chat_dirs = sorted(
        [path for path in results_directory.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    resolved_end_index = end_index if end_index is not None else len(chat_dirs)
    command = [
        sys.executable,
        config.evaluator.evaluation_script,
        "--input_directory",
        str(results_directory),
        "--chat_size",
        config.canonical_tier(tier),
        "--start_index",
        str(start_index),
        "--end_index",
        str(resolved_end_index),
        "--max_workers",
        str(config.runtime.max_workers),
        "--allowed_result_files",
        config.export.result_filename,
    ]
    logger.info("Running BEAM evaluation command=%s cwd=%s", command, beam_root)
    subprocess.run(command, cwd=beam_root, check=True)
    summary = build_evaluation_summary(results_directory, config)
    summary_path = results_directory.parent / "reports" / "{0}-evaluation-summary.json".format(config.canonical_tier(tier))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("BEAM evaluation summary written path=%s", summary_path)
    return summary


def run_beam_report(
    *,
    beam_root: Path,
    results_directory: Path,
    tier: str,
    config: BeamHarnessConfig,
) -> Path:
    """Run the BEAM report export script."""
    evaluation_filename = "{0}{1}".format(config.export.evaluation_prefix, config.export.result_filename)
    output_path = results_directory.parent / "reports" / "{0}-{1}".format(config.canonical_tier(tier), config.export.report_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        config.evaluator.report_script,
        "--evaluation_directory",
        str(results_directory),
        "--row_names",
        evaluation_filename,
        "--output_filename",
        str(output_path),
    ]
    logger.info("Running BEAM report command=%s cwd=%s", command, beam_root)
    subprocess.run(command, cwd=beam_root, check=True)
    logger.info("BEAM report written path=%s", output_path)
    return output_path


def build_evaluation_summary(results_directory: Path, config: BeamHarnessConfig) -> dict[str, Any]:
    """Aggregate numeric signals from evaluation JSON files."""
    evaluation_filename = "{0}{1}".format(config.export.evaluation_prefix, config.export.result_filename)
    chats: dict[str, dict[str, Any]] = {}
    category_scores: dict[str, list[float]] = {}
    for chat_dir in sorted(
        [path for path in results_directory.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    ):
        evaluation_path = chat_dir / evaluation_filename
        if not evaluation_path.exists():
            continue
        payload = json.loads(evaluation_path.read_text())
        chat_summary: dict[str, Any] = {"evaluation_file": str(evaluation_path)}
        for category, items in payload.items():
            scores = [score for score in (_extract_score(item) for item in items) if score is not None]
            if scores:
                average = sum(scores) / len(scores)
                chat_summary[category] = {"count": len(scores), "average_score": average}
                category_scores.setdefault(category, []).extend(scores)
        chats[chat_dir.name] = chat_summary
    overall = {
        category: {"count": len(scores), "average_score": (sum(scores) / len(scores)) if scores else None}
        for category, scores in category_scores.items()
    }
    return {"chats": chats, "overall": overall}


def _extract_score(value: Any) -> float | None:
    """Best-effort numeric score extraction from evaluator output."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "result", "value", "judge_score", "final_score"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
        numeric_values = [item for item in (_extract_score(item) for item in value.values()) if item is not None]
        if len(numeric_values) == 1:
            return numeric_values[0]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
    if isinstance(value, list):
        numeric_values = [item for item in (_extract_score(item) for item in value) if item is not None]
        if len(numeric_values) == 1:
            return numeric_values[0]
    return None
