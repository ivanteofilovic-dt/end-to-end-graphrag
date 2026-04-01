"""Step 5: Generate community reports via Gemini Batch API (level-by-level)."""

from __future__ import annotations

import json
import logging
from typing import Any

from google.cloud import bigquery

from graphrag.batch import client as batch_client
from graphrag.batch.request_builder import build_generation_config, make_request
from graphrag.config import GraphRAGConfig
from graphrag.prompts.community_report import (
    COMMUNITY_REPORT_RESPONSE_SCHEMA,
    COMMUNITY_REPORT_SYSTEM_INSTRUCTION,
    format_community_context,
    format_community_report_prompt,
)
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

COMMUNITIES_REPORT_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "INTEGER"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("summary", "STRING"),
    bigquery.SchemaField("full_content", "STRING"),
    bigquery.SchemaField("rating", "FLOAT"),
    bigquery.SchemaField("rating_explanation", "STRING"),
    bigquery.SchemaField("findings", "JSON"),
    bigquery.SchemaField("entity_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("relationship_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("parent_community_id", "STRING"),
    bigquery.SchemaField("child_community_ids", "STRING", mode="REPEATED"),
]


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 5: Generating community reports")

    communities = bq.read_table_all(cfg, "communities")
    entities = bq.read_table_all(cfg, "entities")
    relationships = bq.read_table_all(cfg, "relationships")

    if not communities:
        logger.warning("No communities found; skipping report generation")
        return

    entity_by_id = {e["id"]: e for e in entities}
    rel_by_id = {r["id"]: r for r in relationships}

    # Group communities by level
    levels: dict[int, list[dict[str, Any]]] = {}
    community_by_id: dict[str, dict[str, Any]] = {}
    for c in communities:
        lvl = c["level"]
        levels.setdefault(lvl, []).append(c)
        community_by_id[c["id"]] = c

    sorted_levels = sorted(levels.keys())
    logger.info("Processing %d levels: %s", len(sorted_levels), sorted_levels)

    # Process bottom-up
    for level in sorted_levels:
        level_communities = levels[level]
        logger.info(
            "Level %d: generating reports for %d communities",
            level, len(level_communities),
        )

        _process_level(
            cfg,
            level=level,
            level_communities=level_communities,
            entity_by_id=entity_by_id,
            rel_by_id=rel_by_id,
            community_by_id=community_by_id,
        )

    # Write all communities (now with reports) to the final table
    output_rows: list[dict[str, Any]] = []
    for c in communities:
        row = dict(c)
        row["findings"] = json.dumps(row.get("findings") or [])
        output_rows.append(row)

    bq.write_rows(cfg, "communities", output_rows, COMMUNITIES_REPORT_SCHEMA)
    logger.info("Step 5 complete: wrote %d community reports", len(output_rows))


def _process_level(
    cfg: GraphRAGConfig,
    *,
    level: int,
    level_communities: list[dict[str, Any]],
    entity_by_id: dict[str, dict],
    rel_by_id: dict[str, dict],
    community_by_id: dict[str, dict],
) -> None:
    """Build request table for one level, run batch job, parse results."""

    gen_config = build_generation_config(
        cfg, response_schema=COMMUNITY_REPORT_RESPONSE_SCHEMA
    )

    request_rows: list[dict[str, Any]] = []
    for community in level_communities:
        c_entities = [
            entity_by_id[eid]
            for eid in community.get("entity_ids", [])
            if eid in entity_by_id
        ]
        c_rels = [
            rel_by_id[rid]
            for rid in community.get("relationship_ids", [])
            if rid in rel_by_id
        ]

        sub_reports: list[dict] | None = None
        child_ids = community.get("child_community_ids") or []
        if child_ids:
            sub_reports = []
            for child_id in child_ids:
                child = community_by_id.get(child_id, {})
                if child.get("summary"):
                    sub_reports.append({
                        "title": child.get("title", ""),
                        "summary": child.get("summary", ""),
                        "rating": child.get("rating"),
                    })

        context = format_community_context(c_entities, c_rels, sub_reports)
        prompt = format_community_report_prompt(context)
        req = make_request(
            prompt, gen_config,
            system_instruction=COMMUNITY_REPORT_SYSTEM_INSTRUCTION,
        )
        request_rows.append({"community_id": community["id"], "request": req})

    if not request_rows:
        return

    req_table = f"report_requests_level_{level}"
    res_table = f"report_results_level_{level}"

    bq.write_batch_request_table(cfg, req_table, request_rows)
    batch_client.run_batch_job(
        cfg, src_table=req_table, dest_table=res_table
    )

    results = batch_client.parse_batch_results(
        cfg, res_table, pass_through_columns=["community_id"]
    )

    for row in results:
        cid = row.get("community_id")
        data = row.get("response_json")
        if not cid or not data:
            continue
        community = community_by_id.get(cid)
        if not community:
            continue

        community["title"] = data.get("title")
        community["summary"] = data.get("summary")
        community["rating"] = data.get("rating")
        community["rating_explanation"] = data.get("rating_explanation")
        community["findings"] = data.get("findings", [])

        parts = [
            f"# {data.get('title', '')}",
            data.get("summary", ""),
            f"Rating: {data.get('rating', 'N/A')} - {data.get('rating_explanation', '')}",
            "## Findings",
        ]
        for f in data.get("findings", []):
            parts.append(f"### {f.get('summary', '')}\n{f.get('explanation', '')}")
        community["full_content"] = "\n\n".join(parts)
