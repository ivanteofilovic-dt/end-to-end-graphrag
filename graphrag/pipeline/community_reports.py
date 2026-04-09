"""Step 4: Generate community reports via Gemini Batch API.

For every community detected by Leiden clustering, collects member nodes and
their internal relationships, submits a batch request to the LLM, and writes
the parsed reports to BigQuery.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from google.cloud import bigquery

from graphrag.batch import client as batch_client
from graphrag.batch.request_builder import build_generation_config, make_request
from graphrag.config import GraphRAGConfig
from graphrag.prompts.community_report import (
    COMMUNITY_REPORT_RESPONSE_SCHEMA,
    format_community_prompt,
    format_system_instruction,
)
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

COMMUNITY_REPORTS_SCHEMA = [
    bigquery.SchemaField("community_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("summary", "STRING"),
    bigquery.SchemaField("findings", "STRING"),
    bigquery.SchemaField("rating", "FLOAT"),
    bigquery.SchemaField("rating_explanation", "STRING"),
]


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 4: Generating community reports")

    _prepare_report_requests(cfg)
    _run_report_batch(cfg)
    _parse_and_write_reports(cfg)

    logger.info("Step 4 complete")


# ── 4a: Build batch requests ────────────────────────────────────────────


def _prepare_report_requests(cfg: GraphRAGConfig) -> None:
    logger.info("Step 4a: Preparing community report requests")

    communities = bq.read_table_all(cfg, "community_info")
    assignments = bq.read_table_all(cfg, "community_assignments")
    nodes = bq.read_table_all(cfg, "merged_nodes")
    relationships = bq.read_table_all(cfg, "merged_relationships")

    if not communities:
        logger.warning("No communities found; skipping report generation")
        return

    node_lookup: dict[str, dict[str, Any]] = {n["id"]: n for n in nodes}

    community_node_ids: dict[str, set[str]] = defaultdict(set)
    for a in assignments:
        community_node_ids[a["community_id"]].add(a["node_id"])

    community_level: dict[str, int] = {
        c["community_id"]: c["level"] for c in communities
    }

    gen_config = build_generation_config(
        cfg, response_schema=COMMUNITY_REPORT_RESPONSE_SCHEMA
    )
    system_instr = format_system_instruction()

    rows: list[dict[str, Any]] = []
    for community in communities:
        cid = community["community_id"]
        member_ids = community_node_ids.get(cid, set())
        if not member_ids:
            continue

        member_nodes = [node_lookup[nid] for nid in member_ids if nid in node_lookup]

        internal_rels = [
            r
            for r in relationships
            if r.get("source_node_id") in member_ids
            and r.get("target_node_id") in member_ids
        ]

        prompt = format_community_prompt(member_nodes, internal_rels, node_lookup)
        req = make_request(prompt, gen_config, system_instruction=system_instr)
        rows.append({
            "community_id": cid,
            "level": str(community_level.get(cid, 0)),
            "request": req,
        })

    if not rows:
        logger.warning("No community report requests to submit")
        return

    bq.write_batch_request_table(cfg, "community_report_requests", rows)
    logger.info("Prepared %d community report requests", len(rows))


# ── 4b: Submit batch job ────────────────────────────────────────────────


def _run_report_batch(cfg: GraphRAGConfig) -> None:
    if not bq.table_exists(cfg, "community_report_requests"):
        logger.warning("No request table found; skipping batch submission")
        return

    logger.info("Step 4b: Submitting community report batch job")
    batch_client.run_batch_job(
        cfg,
        src_table="community_report_requests",
        dest_table="community_report_results",
    )


# ── 4c: Parse results ──────────────────────────────────────────────────


def _parse_and_write_reports(cfg: GraphRAGConfig) -> None:
    if not bq.table_exists(cfg, "community_report_results"):
        logger.warning("No results table found; skipping report parsing")
        return

    logger.info("Step 4c: Parsing community report results")

    results = batch_client.parse_batch_results(
        cfg,
        "community_report_results",
        pass_through_columns=["community_id", "level"],
    )

    report_rows: list[dict[str, Any]] = []
    for row in results:
        data = row.get("response_json")
        if not data:
            cid = row.get("community_id", "?")
            logger.warning("No valid response for community %s", cid)
            continue

        findings = data.get("findings", [])

        report_rows.append({
            "community_id": row.get("community_id", ""),
            "level": int(row.get("level", 0)),
            "title": (data.get("title") or "").strip(),
            "summary": (data.get("summary") or "").strip(),
            "findings": json.dumps(findings, ensure_ascii=False),
            "rating": float(data.get("rating", 0.0)),
            "rating_explanation": (data.get("rating_explanation") or "").strip(),
        })

    if report_rows:
        bq.write_rows(cfg, "community_reports", report_rows, COMMUNITY_REPORTS_SCHEMA)

    logger.info("Generated %d community reports", len(report_rows))
