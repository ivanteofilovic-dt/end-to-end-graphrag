from __future__ import annotations

import json
import logging
import time
from typing import Any

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


def _get_genai_client(cfg: GraphRAGConfig) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=cfg.gcp.project_id,
        location=cfg.gcp.location,
        http_options=HttpOptions(api_version="v1"),
    )


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
) -> JobState:
    """Poll a batch job until it reaches a terminal state."""
    client = _get_genai_client(cfg)

    while True:
        job = client.batches.get(name=job_name)
        logger.info("Batch job %s: state=%s", job_name, job.state)

        if job.state in _TERMINAL_STATES:
            if job.state != JobState.JOB_STATE_SUCCEEDED:
                raise RuntimeError(
                    f"Batch job {job_name} ended with state {job.state}"
                )
            return job.state

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
