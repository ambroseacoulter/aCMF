#!/usr/bin/env python3
"""Send a batch of varied `/v1/process` requests to a running aCMF API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import timedelta

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.time import utc_now


def build_requests(prefix: str) -> list[dict]:
    """Return ten varied process payloads."""
    base_time = utc_now()
    return [
        {
            "user_id": f"{prefix}-user-001",
            "containers": [{"id": f"{prefix}-thread-001", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "Remember that I prefer pytest over unittest.",
                "assistant_response": "Understood. I will use pytest-oriented examples.",
                "occurred_at": (base_time - timedelta(minutes=10)).isoformat(),
                "user_message_id": f"{prefix}-u-001",
                "assistant_message_id": f"{prefix}-a-001",
            },
            "metadata": {"app": "smoke-test", "source": "chat", "trace_id": f"{prefix}-trace-001"},
        },
        {
            "user_id": f"{prefix}-user-001",
            "containers": [{"id": f"{prefix}-project-001", "type": "project"}],
            "scope_policy": {"write_user": "true", "write_global": "false", "write_container": True},
            "turn": {
                "user_message": "I am working on the Neptune migration this week.",
                "assistant_response": "Noted. I will keep the Neptune migration in context.",
                "occurred_at": (base_time - timedelta(minutes=9)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "planner", "model": "manual"},
        },
        {
            "user_id": f"{prefix}-user-002",
            "containers": [],
            "scope_policy": {"write_user": "auto", "write_global": "true", "write_container": False},
            "turn": {
                "user_message": "Across all assistants, always answer with concise bullet points.",
                "assistant_response": "Understood. I will keep responses concise.",
                "occurred_at": (base_time - timedelta(minutes=8)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "policy"},
        },
        {
            "user_id": f"{prefix}-user-003",
            "containers": [{"id": f"{prefix}-case-001", "type": "case"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "The deadline for case Atlas is Friday, and missing it is high risk.",
                "assistant_response": "I will prioritize the Atlas deadline as a high-risk constraint.",
                "occurred_at": (base_time - timedelta(minutes=7)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "ops"},
        },
        {
            "user_id": f"{prefix}-user-004",
            "containers": [{"id": f"{prefix}-repo-001", "type": "repo"}],
            "scope_policy": {"write_user": "true", "write_global": "false", "write_container": True},
            "turn": {
                "user_message": "In this repo, we use SQLAlchemy 2.x and avoid legacy session patterns.",
                "assistant_response": "Understood. I will stick to SQLAlchemy 2.x style.",
                "occurred_at": (base_time - timedelta(minutes=6)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "repo-doc"},
        },
        {
            "user_id": f"{prefix}-user-005",
            "containers": [{"id": f"{prefix}-thread-002", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "I usually work best in the morning between 7 and 11 AM.",
                "assistant_response": "Noted. Morning hours are your preferred work window.",
                "occurred_at": (base_time - timedelta(minutes=5)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "chat"},
        },
        {
            "user_id": f"{prefix}-user-006",
            "containers": [{"id": f"{prefix}-project-002", "type": "project"}],
            "scope_policy": {"write_user": "auto", "write_global": "false", "write_container": True},
            "turn": {
                "user_message": "Please track that Project Cedar depends on the auth service rollout.",
                "assistant_response": "Tracked. Project Cedar is blocked by the auth service rollout.",
                "occurred_at": (base_time - timedelta(minutes=4)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "pm"},
        },
        {
            "user_id": f"{prefix}-user-007",
            "containers": [],
            "scope_policy": {"write_user": "true", "write_global": "false", "write_container": False},
            "turn": {
                "user_message": "I dislike long preambles and want the answer first.",
                "assistant_response": "Understood. I will lead with the answer.",
                "occurred_at": (base_time - timedelta(minutes=3)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "preference"},
        },
        {
            "user_id": f"{prefix}-user-008",
            "containers": [{"id": f"{prefix}-thread-003", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "I met with Annie today to discuss moving the docs to Mintlify.",
                "assistant_response": "Noted. Annie is involved in the Mintlify documentation migration.",
                "occurred_at": (base_time - timedelta(minutes=2)).isoformat(),
            },
            "metadata": {"app": "smoke-test", "source": "meeting-note"},
        },
        {
            "user_id": f"{prefix}-user-009",
            "containers": [{"id": f"{prefix}-thread-004", "type": "thread"}],
            "scope_policy": {"write_user": "auto", "write_global": "auto", "write_container": True},
            "turn": {
                "user_message": "Remember that I switched from text-embedding-3-small to text-embedding-3-large for experiments.",
                "assistant_response": "Understood. I will keep the embedding-model change in mind.",
                "occurred_at": (base_time - timedelta(minutes=1)).isoformat(),
                "user_message_id": f"{prefix}-u-010",
                "assistant_message_id": f"{prefix}-a-010",
            },
            "metadata": {"app": "smoke-test", "source": "experiment-log", "trace_id": f"{prefix}-trace-010"},
        },
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the aCMF API. Default: http://localhost:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--prefix",
        default="smoke",
        help="Prefix used for generated user/container/message ids. Default: smoke",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print response bodies.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the smoke test batch."""
    args = parse_args()
    url = args.base_url.rstrip("/") + "/v1/process"
    payloads = build_requests(args.prefix)
    success_count = 0

    with httpx.Client(timeout=args.timeout) as client:
        for index, payload in enumerate(payloads, start=1):
            response = client.post(url, json=payload)
            ok = response.is_success
            if ok:
                success_count += 1
            summary = {
                "index": index,
                "user_id": payload["user_id"],
                "status_code": response.status_code,
                "ok": ok,
            }
            print(json.dumps(summary))
            if args.pretty:
                try:
                    print(json.dumps(response.json(), indent=2, sort_keys=True))
                except Exception:
                    print(response.text)
            else:
                print(response.text)

    print(
        json.dumps(
            {
                "completed": len(payloads),
                "succeeded": success_count,
                "failed": len(payloads) - success_count,
            }
        )
    )
    return 0 if success_count == len(payloads) else 1


if __name__ == "__main__":
    sys.exit(main())
