from __future__ import annotations

import json
from typing import Any

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq


def build_generation_config(
    cfg: GraphRAGConfig,
    *,
    response_schema: dict | None = None,
) -> dict[str, Any]:
    """Build the ``generationConfig`` portion of a GenerateContentRequest."""
    gen_cfg: dict[str, Any] = {
        "temperature": cfg.llm.temperature,
        "maxOutputTokens": cfg.llm.max_output_tokens,
    }
    if response_schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        gen_cfg["responseSchema"] = response_schema
    return gen_cfg


def make_request(
    prompt: str,
    generation_config: dict[str, Any],
    *,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    """Build a GenerateContentRequest dict for one row."""
    req: dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        "generationConfig": generation_config,
    }
    if system_instruction:
        req["system_instruction"] = {
            "parts": [{"text": system_instruction}],
        }
    return req


def write_request_table(
    cfg: GraphRAGConfig,
    table_name: str,
    rows: list[dict[str, Any]],
) -> str:
    """Write a request table and return the bq:// URI.

    ``rows`` must contain a ``request`` key (dict) and any pass-through string
    columns.
    """
    return bq.write_batch_request_table(cfg, table_name, rows)
