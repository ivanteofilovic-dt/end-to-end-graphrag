"""Step 2: Extract entities and relationships via Gemini Batch API."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
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
from graphrag.prompts.summarization import (
    SUMMARIZATION_RESPONSE_SCHEMA,
    SUMMARIZATION_SYSTEM_INSTRUCTION,
    format_entity_summarization_prompt,
    format_relationship_summarization_prompt,
)
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

ENTITIES_RAW_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("type", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
]

RELATIONSHIPS_RAW_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("target", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("weight", "FLOAT"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
]


def run(cfg: GraphRAGConfig, *, max_rows: int | None = None) -> None:
    logger.info("Step 2: Extracting graph from documents")

    _prepare_extraction_requests(cfg, max_rows=max_rows)
    _run_extraction_batch(cfg)
    entities_merged, rels_merged = _parse_and_merge(cfg)
    _run_summarization(cfg, entities_merged, rels_merged)

    logger.info("Step 2 complete")


# ── Sub-step 2a: Prepare extraction request table ────────────────────────


def _prepare_extraction_requests(
    cfg: GraphRAGConfig, *, max_rows: int | None = None
) -> None:
    logger.info("Step 2a: Preparing extraction requests")
    docs = bq.read_table_all(cfg, "documents", columns=["id", "raw_content"])
    if max_rows is not None:
        docs = docs[:max_rows]
        logger.info("Limiting extraction to %d documents (--max-rows)", max_rows)

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


# ── Sub-step 2b: Submit extraction batch job ─────────────────────────────


def _run_extraction_batch(cfg: GraphRAGConfig) -> None:
    logger.info("Step 2b: Submitting extraction batch job")
    batch_client.run_batch_job(
        cfg,
        src_table="extraction_requests",
        dest_table="extraction_results",
    )


# ── Sub-step 2c: Parse results and merge ─────────────────────────────────

EntityKey = tuple[str, str]  # (title_upper, type)
RelKey = tuple[str, str]     # (source_upper, target_upper)


def _parse_and_merge(
    cfg: GraphRAGConfig,
) -> tuple[dict[EntityKey, dict], dict[RelKey, dict]]:
    """Parse extraction results and merge entities/relationships by key."""
    logger.info("Step 2c: Parsing extraction results and merging")

    results = batch_client.parse_batch_results(
        cfg, "extraction_results", pass_through_columns=["document_id"]
    )

    entities_map: dict[EntityKey, dict[str, Any]] = defaultdict(
        lambda: {"descriptions": [], "document_ids": set(), "type": ""}
    )
    rels_map: dict[RelKey, dict[str, Any]] = defaultdict(
        lambda: {"descriptions": [], "document_ids": set(), "weights": []}
    )

    for row in results:
        data = row.get("response_json")
        if not data:
            continue
        doc_id = row.get("document_id", "")

        for ent in data.get("entities", []):
            name = ent.get("name", "").strip().upper()
            etype = ent.get("type", "CONCEPT").strip().upper()
            desc = ent.get("description", "")
            if not name:
                continue
            key: EntityKey = (name, etype)
            entities_map[key]["descriptions"].append(desc)
            entities_map[key]["document_ids"].add(doc_id)
            entities_map[key]["type"] = etype

        for rel in data.get("relationships", []):
            source = rel.get("source", "").strip().upper()
            target = rel.get("target", "").strip().upper()
            desc = rel.get("description", "")
            weight = float(rel.get("weight", 1.0))
            if not source or not target:
                continue
            rkey: RelKey = (source, target)
            rels_map[rkey]["descriptions"].append(desc)
            rels_map[rkey]["document_ids"].add(doc_id)
            rels_map[rkey]["weights"].append(weight)

    # Convert sets to lists for serialization
    for v in entities_map.values():
        v["document_ids"] = list(v["document_ids"])
    for v in rels_map.values():
        v["document_ids"] = list(v["document_ids"])

    logger.info(
        "Merged into %d unique entities and %d unique relationships",
        len(entities_map), len(rels_map),
    )
    return dict(entities_map), dict(rels_map)


# ── Sub-steps 2d+2e: Summarize multi-description entries ─────────────────


def _run_summarization(
    cfg: GraphRAGConfig,
    entities_merged: dict[EntityKey, dict],
    rels_merged: dict[RelKey, dict],
) -> None:
    """If any entity/relationship has multiple descriptions, run a second
    batch job to summarize them. Then write final entities_raw and
    relationships_raw tables."""

    needs_summarization: list[dict[str, Any]] = []

    # Check entities
    for (name, etype), data in entities_merged.items():
        if len(data["descriptions"]) > 1:
            prompt = format_entity_summarization_prompt(
                name, etype, data["descriptions"]
            )
            pass_key = f"entity::{name}::{etype}"
            gen_config = build_generation_config(
                cfg, response_schema=SUMMARIZATION_RESPONSE_SCHEMA
            )
            req = make_request(
                prompt, gen_config,
                system_instruction=SUMMARIZATION_SYSTEM_INSTRUCTION,
            )
            needs_summarization.append({
                "entity_or_rel_key": pass_key,
                "request": req,
            })

    # Check relationships
    for (source, target), data in rels_merged.items():
        if len(data["descriptions"]) > 1:
            prompt = format_relationship_summarization_prompt(
                source, target, data["descriptions"]
            )
            pass_key = f"rel::{source}::{target}"
            gen_config = build_generation_config(
                cfg, response_schema=SUMMARIZATION_RESPONSE_SCHEMA
            )
            req = make_request(
                prompt, gen_config,
                system_instruction=SUMMARIZATION_SYSTEM_INSTRUCTION,
            )
            needs_summarization.append({
                "entity_or_rel_key": pass_key,
                "request": req,
            })

    # Run summarization batch if needed
    summaries: dict[str, str] = {}
    if needs_summarization:
        logger.info(
            "Step 2d: Preparing %d summarization requests",
            len(needs_summarization),
        )
        bq.write_batch_request_table(
            cfg, "summarization_requests", needs_summarization
        )
        logger.info("Step 2e: Submitting summarization batch job")
        batch_client.run_batch_job(
            cfg,
            src_table="summarization_requests",
            dest_table="summarization_results",
        )
        results = batch_client.parse_batch_results(
            cfg,
            "summarization_results",
            pass_through_columns=["entity_or_rel_key"],
        )
        for row in results:
            pk = row.get("entity_or_rel_key", "")
            data = row.get("response_json")
            if data and "description" in data:
                summaries[pk] = data["description"]
    else:
        logger.info("No multi-description entries; skipping summarization")

    # Build final entities
    entity_rows: list[dict[str, Any]] = []
    for (name, etype), data in entities_merged.items():
        pass_key = f"entity::{name}::{etype}"
        if pass_key in summaries:
            desc = summaries[pass_key]
        elif data["descriptions"]:
            desc = data["descriptions"][0]
        else:
            desc = ""

        entity_rows.append({
            "id": str(uuid.uuid4()),
            "title": name,
            "type": etype,
            "description": desc,
            "document_ids": data["document_ids"],
        })

    # Build final relationships
    rel_rows: list[dict[str, Any]] = []
    for (source, target), data in rels_merged.items():
        pass_key = f"rel::{source}::{target}"
        if pass_key in summaries:
            desc = summaries[pass_key]
        elif data["descriptions"]:
            desc = data["descriptions"][0]
        else:
            desc = ""

        avg_weight = (
            sum(data["weights"]) / len(data["weights"])
            if data["weights"]
            else 1.0
        )

        rel_rows.append({
            "id": str(uuid.uuid4()),
            "source": source,
            "target": target,
            "description": desc,
            "weight": avg_weight,
            "document_ids": data["document_ids"],
        })

    bq.write_rows(cfg, "entities_raw", entity_rows, ENTITIES_RAW_SCHEMA)
    bq.write_rows(cfg, "relationships_raw", rel_rows, RELATIONSHIPS_RAW_SCHEMA)
    logger.info(
        "Wrote %d entities and %d relationships to raw tables",
        len(entity_rows), len(rel_rows),
    )
