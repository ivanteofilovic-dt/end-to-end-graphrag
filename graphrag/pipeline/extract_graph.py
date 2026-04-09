"""Step 1: Extract typed nodes and relationships via Gemini Batch API."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from google.cloud import bigquery

from graphrag.batch import client as batch_client
from graphrag.batch.request_builder import build_generation_config, make_request
from graphrag.config import GraphRAGConfig
from graphrag.prompts.extraction import (
    EXTRACTION_RESPONSE_SCHEMA,
    format_extraction_prompt,
    format_system_instruction,
)
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

RAW_NODES_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("document_id", "STRING"),
    bigquery.SchemaField("node_type", "STRING"),
    bigquery.SchemaField("name", "STRING"),
    # Call
    bigquery.SchemaField("call_category", "STRING"),
    bigquery.SchemaField("call_outcome", "STRING"),
    bigquery.SchemaField("call_timestamp", "STRING"),
    # Customer
    bigquery.SchemaField("customer_id", "STRING"),
    bigquery.SchemaField("customer_type", "STRING"),
    bigquery.SchemaField("overall_sentiment", "STRING"),
    # Agent
    bigquery.SchemaField("agent_id", "STRING"),
    bigquery.SchemaField("agent_role", "STRING"),
    # Problem
    bigquery.SchemaField("issue_type", "STRING"),
    bigquery.SchemaField("severity", "STRING"),
    bigquery.SchemaField("issue_description", "STRING"),
    # Product
    bigquery.SchemaField("product_name", "STRING"),
    bigquery.SchemaField("product_type", "STRING"),
    # Service
    bigquery.SchemaField("service_name", "STRING"),
    bigquery.SchemaField("service_category", "STRING"),
    # Solution
    bigquery.SchemaField("solution_type", "STRING"),
    bigquery.SchemaField("resolution_status", "STRING"),
    # Feedback
    bigquery.SchemaField("feedback_type", "STRING"),
    bigquery.SchemaField("feedback_sentiment", "STRING"),
    # Generic
    bigquery.SchemaField("description", "STRING"),
]

RAW_RELATIONSHIPS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("document_id", "STRING"),
    bigquery.SchemaField("relationship_type", "STRING"),
    bigquery.SchemaField("source_name", "STRING"),
    bigquery.SchemaField("target_name", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("weight", "FLOAT"),
]


def run(cfg: GraphRAGConfig, *, max_rows: int | None = None) -> None:
    logger.info("Step 1: Extracting graph from source table")

    _prepare_extraction_requests(cfg, max_rows=max_rows)
    _run_extraction_batch(cfg)
    _parse_and_write_raw(cfg)

    logger.info("Step 1 complete")


# ── 1a: Prepare extraction request table ─────────────────────────────────


def _prepare_extraction_requests(
    cfg: GraphRAGConfig, *, max_rows: int | None = None
) -> None:
    logger.info("Step 1a: Preparing extraction requests")

    client = bq.get_client(cfg)
    limit_clause = f" LIMIT {max_rows}" if max_rows else ""
    # TODO: missing timestamp
    query = (
        f"SELECT data_id, full_conversation"
        f" FROM `{cfg.source_table_fqn()}`"
        f"{limit_clause}"
    )
    if max_rows:
        logger.info("Limiting extraction to %d documents (--max-rows)", max_rows)

    job = client.query(query)
    docs = [{"id": str(row["data_id"]), "raw_content": row["full_conversation"] or ""} for row in job.result()]
    gen_config = build_generation_config(
        cfg, response_schema=EXTRACTION_RESPONSE_SCHEMA
    )
    system_instr = format_system_instruction()

    rows: list[dict[str, Any]] = []
    for doc in docs:
        prompt = format_extraction_prompt(doc["raw_content"])
        req = make_request(prompt, gen_config, system_instruction=system_instr)
        rows.append({"document_id": doc["id"], "request": req})

    bq.write_batch_request_table(cfg, "extraction_requests", rows)
    logger.info("Prepared %d extraction requests", len(rows))


# ── 1b: Submit batch job ─────────────────────────────────────────────────


def _run_extraction_batch(cfg: GraphRAGConfig) -> None:
    logger.info("Step 1b: Submitting extraction batch job")
    batch_client.run_batch_job(
        cfg,
        src_table="extraction_requests",
        dest_table="extraction_results",
    )


# ── 1c: Parse results into raw_nodes / raw_relationships ────────────────

_ATTR_MAP = {
    "Call": {
        "call_category": "call_category",
        "call_outcome": "call_outcome",
        "timestamp": "call_timestamp",
    },
    "Customer": {
        "customer_id": "customer_id",
        "customer_type": "customer_type",
        "overall_sentiment": "overall_sentiment",
    },
    "Agent": {
        "agent_id": "agent_id",
        "role": "agent_role",
    },
    "Problem": {
        "issue_type": "issue_type",
        "severity": "severity",
        "description": "issue_description",
    },
    "Product": {
        "product_name": "product_name",
        "product_type": "product_type",
    },
    "Service": {
        "service_name": "service_name",
        "service_category": "service_category",
    },
    "Solution": {
        "solution_type": "solution_type",
        "resolution_status": "resolution_status",
    },
    "Feedback": {
        "feedback_type": "feedback_type",
        "sentiment": "feedback_sentiment",
    },
}


def _parse_and_write_raw(cfg: GraphRAGConfig) -> None:
    logger.info("Step 1c: Parsing extraction results")

    results = batch_client.parse_batch_results(
        cfg, "extraction_results", pass_through_columns=["document_id"]
    )

    node_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []

    for row in results:
        data = row.get("response_json")
        if not data:
            continue
        doc_id = row.get("document_id", "")

        for node in data.get("nodes", []):
            name = (node.get("name") or "").strip().upper()
            node_type = (node.get("node_type") or "").strip()
            if not name or not node_type:
                continue

            attrs = node.get("attributes") or {}
            flat: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "node_type": node_type,
                "name": name,
            }

            attr_mapping = _ATTR_MAP.get(node_type, {})
            for src_key, dest_key in attr_mapping.items():
                val = attrs.get(src_key)
                if val:
                    flat[dest_key] = str(val).strip()

            if node_type not in ("Problem",):
                desc = attrs.get("description")
                if desc:
                    flat["description"] = str(desc).strip()

            node_rows.append(flat)

        for rel in data.get("relationships", []):
            source = (rel.get("source") or "").strip().upper()
            target = (rel.get("target") or "").strip().upper()
            rel_type = (rel.get("relationship_type") or "").strip()
            if not source or not target or not rel_type:
                continue

            rel_rows.append({
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "relationship_type": rel_type,
                "source_name": source,
                "target_name": target,
                "description": (rel.get("description") or "").strip(),
                "weight": float(rel.get("weight", 1.0)),
            })

    bq.write_rows(cfg, "raw_nodes", node_rows, RAW_NODES_SCHEMA)
    bq.write_rows(cfg, "raw_relationships", rel_rows, RAW_RELATIONSHIPS_SCHEMA)
    logger.info(
        "Parsed %d raw nodes and %d raw relationships",
        len(node_rows), len(rel_rows),
    )
