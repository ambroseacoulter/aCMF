"""CLI entrypoint for the BEAM harness."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.benchmarks.beam.config import BeamHarnessConfig
from app.benchmarks.beam.evaluator import run_beam_evaluation, run_beam_report
from app.benchmarks.beam.runner import run_tier


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the aCMF BEAM benchmark harness.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[3] / "benchmarks" / "beam" / "canonical-config.json"),
        help="Path to the benchmark config JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal log verbosity for the harness. Default: INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Ingest BEAM chats and export aCMF answers.")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--image-ref", default=None, help="Docker image tag or digest used for the aCMF runtime.")

    eval_parser = subparsers.add_parser("evaluate", help="Run the official BEAM evaluator.")
    _add_common_arguments(eval_parser)

    report_parser = subparsers.add_parser("report", help="Run the official BEAM report exporter.")
    _add_common_arguments(report_parser)
    return parser


def main() -> int:
    """Run the selected CLI subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)
    config = BeamHarnessConfig.load(Path(args.config))
    beam_root = Path(args.beam_root).resolve()
    output_directory = Path(args.output_dir).resolve()
    if args.command == "run":
        result = run_tier(
            beam_root=beam_root,
            output_directory=output_directory,
            tier=args.tier,
            run_id=args.run_id,
            config=config,
            api_base_url=args.api_base_url,
            image_ref=args.image_ref,
        )
        print(result.results_directory)
        return 0
    results_directory = output_directory / args.run_id / "results" / config.canonical_tier(args.tier)
    if args.command == "evaluate":
        summary = run_beam_evaluation(
            beam_root=beam_root,
            results_directory=results_directory,
            tier=args.tier,
            config=config,
        )
        print(results_directory.parent / "reports" / "{0}-evaluation-summary.json".format(config.canonical_tier(args.tier)))
        print(summary["overall"])
        return 0
    report_path = run_beam_report(
        beam_root=beam_root,
        results_directory=results_directory,
        tier=args.tier,
        config=config,
    )
    print(report_path)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for all subcommands."""
    parser.add_argument("--beam-root", required=True, help="Path to the local BEAM repository checkout.")
    parser.add_argument("--tier", required=True, help="BEAM tier: 100K, 500K, 1M, or 10M.")
    parser.add_argument("--output-dir", required=True, help="Directory where benchmark artifacts will be written.")
    parser.add_argument("--run-id", required=True, help="Stable run identifier used for artifact names and user ids.")
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Base URL for the running aCMF API. Defaults to the value in the config file.",
    )


def _configure_logging(level: str) -> None:
    """Configure simple terminal logging for the harness."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
