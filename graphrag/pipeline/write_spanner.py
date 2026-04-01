"""Step 4: Write merged knowledge graph to Spanner Graph."""

from __future__ import annotations

import logging
from typing import Any

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq
from graphrag.storage import spanner

logger = logging.getLogger(__name__)


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 4: Writing merged graph to Spanner")

    nodes = bq.read_table_all(cfg, "merged_nodes")
    relationships = bq.read_table_all(cfg, "merged_relationships")

    if not nodes:
        logger.warning("No merged nodes found; skipping Spanner write")
        return

    logger.info("Creating Spanner schema")
    spanner.create_schema(cfg)

    node_rows = _prepare_node_rows(nodes)
    rel_rows = _prepare_relationship_rows(relationships)

    logger.info(
        "Writing to Spanner: %d nodes, %d relationships",
        len(node_rows), len(rel_rows),
    )

    spanner.bulk_write_nodes(cfg, node_rows)
    spanner.bulk_write_relationships(cfg, rel_rows)

    logger.info("Step 4 complete: Spanner sync finished")


def _prepare_node_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for n in nodes:
        rows.append({
            "id": n["id"],
            "node_type": n.get("node_type", ""),
            "name": n.get("name", ""),
            # Call
            "call_category": n.get("call_category"),
            "call_outcome": n.get("call_outcome"),
            "call_timestamp": n.get("call_timestamp"),
            # Customer
            "customer_id": n.get("customer_id"),
            "customer_type": n.get("customer_type"),
            "overall_sentiment": n.get("overall_sentiment"),
            # Agent
            "agent_id": n.get("agent_id"),
            "agent_role": n.get("agent_role"),
            # Problem
            "issue_type": n.get("issue_type"),
            "severity": n.get("severity"),
            "issue_description": n.get("issue_description"),
            # Product
            "product_name": n.get("product_name"),
            "product_type": n.get("product_type"),
            # Service
            "service_name": n.get("service_name"),
            "service_category": n.get("service_category"),
            # Solution
            "solution_type": n.get("solution_type"),
            "resolution_status": n.get("resolution_status"),
            # Feedback
            "feedback_type": n.get("feedback_type"),
            "feedback_sentiment": n.get("feedback_sentiment"),
            # Metadata
            "document_ids": n.get("document_ids", []),
            "description": n.get("description"),
        })
    return rows


def _prepare_relationship_rows(
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for r in relationships:
        rows.append({
            "id": r["id"],
            "relationship_type": r.get("relationship_type", ""),
            "source_node_id": r.get("source_node_id", ""),
            "target_node_id": r.get("target_node_id", ""),
            "description": r.get("description"),
            "weight": r.get("weight", 1.0),
            "document_ids": r.get("document_ids", []),
        })
    return rows
