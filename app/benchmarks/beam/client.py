"""HTTP client helpers for the BEAM harness."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.benchmarks.beam.config import BeamHarnessConfig


@dataclass
class JobStatusHistory:
    """Polled status entries for one background job."""

    job_id: str
    statuses: list[dict[str, Any]]


class BeamAPIClient:
    """Minimal aCMF API client for benchmark execution."""

    def __init__(self, config: BeamHarnessConfig, *, base_url: str | None = None) -> None:
        self.config = config
        self.base_url = (base_url or config.runtime.api_base_url).rstrip("/")
        self.client = httpx.Client(timeout=config.runtime.request_timeout_seconds)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def healthcheck(self) -> dict[str, Any]:
        """Call the API health endpoint."""
        return self._request_json("GET", "/health")

    def process_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit one turn for async processing."""
        return self._request_json("POST", "/v1/process", json=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Fetch one background job status."""
        return self._request_json("GET", "/v1/jobs/{0}".format(job_id))

    def answer_deep_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask one deep-memory question."""
        return self._request_json("POST", "/v1/deep-memory", json=payload)

    def wait_for_job(self, job_id: str) -> JobStatusHistory:
        """Poll one job until success or failure."""
        deadline = time.monotonic() + self.config.runtime.job_timeout_seconds
        statuses: list[dict[str, Any]] = []
        while True:
            status = self.get_job(job_id)
            statuses.append(status)
            if status["status"] in {"succeeded", "failed"}:
                return JobStatusHistory(job_id=job_id, statuses=statuses)
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for job {0}".format(job_id))
            time.sleep(self.config.runtime.job_poll_interval_seconds)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send one request with transport-level retries."""
        last_error: Exception | None = None
        url = self.base_url + path
        for attempt in range(self.config.runtime.transport_retries + 1):
            try:
                response = self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return dict(response.json())
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self.config.runtime.transport_retries:
                    break
                time.sleep(self.config.runtime.transport_backoff_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without a transport error: {0} {1}".format(method, url))
