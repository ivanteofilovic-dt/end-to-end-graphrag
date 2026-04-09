from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions, JobState

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

_POLL_INTERVAL_SECONDS = 30
_MAX_CONSECUTIVE_ERRORS = 10
_MAX_BACKOFF_SECONDS = 300


def _get_genai_client(cfg: GraphRAGConfig) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=cfg.gcp.project_id,
        location=cfg.gcp.location,
        http_options=HttpOptions(api_version="v1"),
    )


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient network error."""
    return isinstance(exc, (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.NetworkError,
        ConnectionError,
        OSError,
    ))


def submit_batch_job(
    cfg: GraphRAGConfig,
    *,
    src_table: str,
    dest_table: str,
    model: str | None = None,
) -> str:
    """Submit a Gemini Batch Prediction job with BigQuery I/O.

    Args:
        cfg: Pipeline configuration.
        src_table: BQ table name (within the intermediate dataset) containing
            a ``request`` column with GenerateContentRequest JSON.
        dest_table: BQ table name for output.
        model: Override the model from config.

    Returns:
        The batch job resource name (for polling).
    """
    client = _get_genai_client(cfg)
    src_uri = f"bq://{cfg.table_fqn(src_table)}"
    dest_uri = f"bq://{cfg.table_fqn(dest_table)}"
    model_name = model or cfg.llm.model

    logger.info(
        "Submitting batch job: model=%s, src=%s, dest=%s",
        model_name, src_uri, dest_uri,
    )

    job = client.batches.create(
        model=model_name,
        src=src_uri,
        config=CreateBatchJobConfig(dest=dest_uri),
    )
    logger.info("Batch job submitted: %s (state=%s)", job.name, job.state)
    return job.name


def poll_until_done(
    cfg: GraphRAGConfig,
    job_name: str,
    *,
    poll_interval: int = _POLL_INTERVAL_SECONDS,
    max_retries: int = _MAX_CONSECUTIVE_ERRORS,
) -> JobState:
    """Poll a batch job until it reaches a terminal state.

    Retries with exponential backoff on transient network errors so that
    a momentary DNS or connectivity blip does not kill a multi-hour run.
    """
    client = _get_genai_client(cfg)
    consecutive_errors = 0

    while True:
        try:
            job = client.batches.get(name=job_name)
            consecutive_errors = 0
            logger.info("Batch job %s: state=%s", job_name, job.state)

            if job.state in _TERMINAL_STATES:
                if job.state != JobState.JOB_STATE_SUCCEEDED:
                    raise RuntimeError(
                        f"Batch job {job_name} ended with state {job.state}"
                    )
                return job.state

        except Exception as exc:
            if not _is_transient(exc):
                raise
            consecutive_errors += 1
            if consecutive_errors > max_retries:
                raise RuntimeError(
                    f"Giving up after {max_retries} consecutive transient "
                    f"errors polling {job_name}. Last error: {exc}"
                ) from exc
            backoff = min(
                poll_interval * (2 ** (consecutive_errors - 1)),
                _MAX_BACKOFF_SECONDS,
            )
            logger.warning(
                "Transient error polling %s (attempt %d/%d): %s — retrying in %ds",
                job_name, consecutive_errors, max_retries, exc, backoff,
            )
            time.sleep(backoff)
            continue

        time.sleep(poll_interval)


def run_batch_job(
    cfg: GraphRAGConfig,
    *,
    src_table: str,
    dest_table: str,
    model: str | None = None,
    poll_interval: int = _POLL_INTERVAL_SECONDS,
) -> None:
    """Submit a batch job and block until it succeeds."""
    job_name = submit_batch_job(
        cfg, src_table=src_table, dest_table=dest_table, model=model
    )
    poll_until_done(cfg, job_name, poll_interval=poll_interval)
    logger.info("Batch job completed: %s", job_name)


# ── Batch job management ────────────────────────────────────────────────


def list_batch_jobs(cfg: GraphRAGConfig, *, limit: int = 20) -> list:
    """Return the most recent batch jobs for the configured project."""
    client = _get_genai_client(cfg)
    return list(client.batches.list(config={"page_size": limit}))


def get_batch_job(cfg: GraphRAGConfig, job_name: str) -> Any:
    """Fetch a single batch job by resource name."""
    client = _get_genai_client(cfg)
    return client.batches.get(name=job_name)


def parse_batch_results(
    cfg: GraphRAGConfig,
    result_table: str,
    *,
    pass_through_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read results from a batch output table and parse the response JSON.

    Returns a list of dicts with:
      - All pass-through columns
      - ``response_text``: the model's text output
      - ``response_json``: parsed JSON (if the output is valid JSON), else None
      - ``status``: error status string if the row failed, else None
    """
    columns = list(pass_through_columns or []) + ["response", "status"]
    rows = bq.read_table_all(cfg, result_table, columns=columns)

    parsed: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {}
        for col in (pass_through_columns or []):
            entry[col] = row.get(col)

        status = row.get("status")
        response = row.get("response")

        if status:
            entry["status"] = status
            entry["response_text"] = None
            entry["response_json"] = None
            logger.warning("Row failed: %s", status)
        else:
            text = _extract_text_from_response(response)
            entry["status"] = None
            entry["response_text"] = text
            try:
                entry["response_json"] = json.loads(text) if text else None
            except (json.JSONDecodeError, TypeError):
                entry["response_json"] = None
                logger.warning("Could not parse response as JSON: %s", text[:200] if text else "")

        parsed.append(entry)

    return parsed


def _extract_text_from_response(response: Any) -> str | None:
    """Extract the text from a Gemini batch response column."""
    if response is None:
        return None

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return response

    try:
        candidates = response.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        return parts[0].get("text")
    except (AttributeError, IndexError, KeyError):
        return None
