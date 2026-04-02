"""Step 3: Write merged knowledge graph to Spanner Graph.

Uses a schematized graph model with typed property columns and DYNAMIC LABEL
for per-type labeling. The ``KnowledgeGraph`` property graph is queryable via
GQL, e.g.:

    GRAPH KnowledgeGraph
    MATCH (c:Customer)-[r:Initiated]->(call:Call)
    RETURN c.name, call.call_category
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq
from graphrag.storage import spanner

logger = logging.getLogger(__name__)


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 3: Writing merged graph to Spanner")

    nodes = bq.read_table_all(cfg, "merged_nodes")
    relationships = bq.read_table_all(cfg, "merged_relationships")

    if not nodes:
        logger.warning("No merged nodes found; skipping Spanner write")
        return

    logger.info("Creating Spanner graph schema")
    spanner.create_schema(cfg)

    node_rows = _prepare_node_rows(nodes)
    edge_rows = _prepare_edge_rows(relationships)

    logger.info(
        "Writing to Spanner graph: %d nodes, %d edges",
        len(node_rows), len(edge_rows),
    )

    spanner.bulk_write_nodes(cfg, node_rows)
    spanner.bulk_write_edges(cfg, edge_rows)

    logger.info("Step 3 complete: Spanner graph sync finished")


def _prepare_node_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert merged nodes into Nodes table rows with typed columns."""
    rows: list[dict[str, Any]] = []
    for n in nodes:
        rows.append({
            "id": n["id"],
            "node_type": (n.get("node_type") or "unknown").lower(),
            "name": n.get("name", ""),
            "call_category": n.get("call_category"),
            "call_outcome": n.get("call_outcome"),
            "call_timestamp": n.get("call_timestamp"),
            "customer_id": n.get("customer_id"),
            "customer_type": n.get("customer_type"),
            "overall_sentiment": n.get("overall_sentiment"),
            "agent_id": n.get("agent_id"),
            "agent_role": n.get("agent_role"),
            "issue_type": n.get("issue_type"),
            "severity": n.get("severity"),
            "issue_description": n.get("issue_description"),
            "product_name": n.get("product_name"),
            "product_type": n.get("product_type"),
            "service_name": n.get("service_name"),
            "service_category": n.get("service_category"),
            "solution_type": n.get("solution_type"),
            "resolution_status": n.get("resolution_status"),
            "feedback_type": n.get("feedback_type"),
            "feedback_sentiment": n.get("feedback_sentiment"),
            "document_ids": n.get("document_ids", []),
            "description": n.get("description"),
        })
    return rows


def _prepare_edge_rows(
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert merged relationships into Relationships table rows.

    Column mapping for interleaved edge table:
      source_node_id -> id  (matches Nodes PK for interleaving)
      relationship id -> edge_id
    """
    rows: list[dict[str, Any]] = []
    for r in relationships:
        rows.append({
            "id": r.get("source_node_id", ""),
            "target_node_id": r.get("target_node_id", ""),
            "edge_id": r["id"],
            "relationship_type": (r.get("relationship_type") or "unknown").lower(),
            "description": r.get("description"),
            "weight": r.get("weight", 1.0),
            "document_ids": r.get("document_ids", []),
        })
    return rows
