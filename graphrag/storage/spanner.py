"""Spanner Graph storage: schema creation, bulk writes, and property graph DDL.

Uses a schematized graph model with typed columns for node/edge properties
and DYNAMIC LABEL for flexible per-type labeling. The Relationships table
is interleaved in Nodes for efficient forward edge traversal.
"""

from __future__ import annotations

import logging
from typing import Any

from google.cloud import spanner

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS Nodes (
        id STRING(64) NOT NULL,
        node_type STRING(64) NOT NULL,
        name STRING(MAX),
        -- Call
        call_category STRING(256),
        call_outcome STRING(256),
        call_timestamp STRING(256),
        -- Customer
        customer_id STRING(256),
        customer_type STRING(256),
        overall_sentiment STRING(64),
        -- Agent
        agent_id STRING(256),
        agent_role STRING(256),
        -- Problem
        issue_type STRING(256),
        severity STRING(64),
        issue_description STRING(MAX),
        -- Product
        product_name STRING(256),
        product_type STRING(256),
        -- Service
        service_name STRING(256),
        service_category STRING(256),
        -- Solution
        solution_type STRING(256),
        resolution_status STRING(256),
        -- Feedback
        feedback_type STRING(256),
        feedback_sentiment STRING(64),
        -- Metadata
        document_ids ARRAY<STRING(64)>,
        description STRING(MAX)
    ) PRIMARY KEY (id)
    """,
    """
    CREATE TABLE IF NOT EXISTS Relationships (
        id STRING(64) NOT NULL,
        target_node_id STRING(64) NOT NULL,
        edge_id STRING(64) NOT NULL,
        relationship_type STRING(64) NOT NULL,
        description STRING(MAX),
        weight FLOAT64,
        document_ids ARRAY<STRING(64)>,
        CONSTRAINT FK_TargetNode FOREIGN KEY (target_node_id)
            REFERENCES Nodes(id) NOT ENFORCED
    ) PRIMARY KEY (id, target_node_id, edge_id),
        INTERLEAVE IN PARENT Nodes ON DELETE CASCADE
    """,
]

INDEX_STATEMENTS = [
    """
    CREATE INDEX IF NOT EXISTS NodesByType
        ON Nodes(node_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS IdxReverseEdge
        ON Relationships(target_node_id, id, edge_id)
        STORING (relationship_type, description, weight, document_ids),
        INTERLEAVE IN Nodes
    """,
    """
    CREATE INDEX IF NOT EXISTS RelsByType
        ON Relationships(relationship_type)
    """,
]

PROPERTY_GRAPH_DDL = """
    CREATE OR REPLACE PROPERTY GRAPH KnowledgeGraph
        NODE TABLES (
            Nodes
                KEY (id)
                PROPERTIES ALL COLUMNS EXCEPT (node_type)
                DYNAMIC LABEL (node_type)
        )
        EDGE TABLES (
            Relationships
                KEY (edge_id)
                SOURCE KEY (id) REFERENCES Nodes(id)
                DESTINATION KEY (target_node_id) REFERENCES Nodes(id)
                PROPERTIES (description, weight, document_ids)
                DYNAMIC LABEL (relationship_type)
        )
"""


_GRAPH_TABLES = ["Relationships", "GraphEdge", "Nodes", "GraphNode"]


def get_instance(cfg: GraphRAGConfig) -> spanner.Client:
    return spanner.Client(project=cfg.gcp.project_id)


def _discover_existing_objects(database, table_names: set[str]) -> tuple[list[str], list[str]]:
    """Query INFORMATION_SCHEMA for property graphs and indexes to drop.

    Returns (property_graph_names, index_names).
    """
    graphs: list[str] = []
    indexes: list[str] = []
    try:
        with database.snapshot() as snapshot:
            for row in snapshot.execute_sql(
                "SELECT PROPERTY_GRAPH_NAME "
                "FROM INFORMATION_SCHEMA.PROPERTY_GRAPHS"
            ):
                graphs.append(row[0])

            for row in snapshot.execute_sql(
                "SELECT INDEX_NAME, TABLE_NAME "
                "FROM INFORMATION_SCHEMA.INDEXES "
                "WHERE INDEX_TYPE != 'PRIMARY_KEY' "
                "AND SPANNER_IS_MANAGED = FALSE"
            ):
                if row[1] in table_names:
                    indexes.append(row[0])
    except Exception:
        logger.debug("Could not query INFORMATION_SCHEMA, skipping object discovery")
    return graphs, indexes


def create_schema(cfg: GraphRAGConfig) -> None:
    """Drop and recreate Spanner graph tables, indexes, and property graph.

    Data in Spanner is always derived from BigQuery, so a full recreate
    is safe and avoids schema-migration issues between runs.

    Property graphs, indexes, and tables are discovered dynamically via
    INFORMATION_SCHEMA so we never miss leftovers from previous schema versions.
    """
    client = get_instance(cfg)
    instance = client.instance(cfg.spanner.instance_id)
    database = instance.database(cfg.spanner.database_id)

    graphs, indexes = _discover_existing_objects(database, set(_GRAPH_TABLES))

    cleanup_ddl: list[str] = []

    for graph in graphs:
        cleanup_ddl.append(f"DROP PROPERTY GRAPH IF EXISTS {graph}")

    for idx in indexes:
        cleanup_ddl.append(f"DROP INDEX IF EXISTS {idx}")

    for table in _GRAPH_TABLES:
        cleanup_ddl.append(f"DROP TABLE IF EXISTS {table}")

    logger.info("Dropping existing Spanner graph schema (%d statements)...", len(cleanup_ddl))
    database.update_ddl(cleanup_ddl).result()

    create_ddl = DDL_STATEMENTS + INDEX_STATEMENTS + [PROPERTY_GRAPH_DDL]
    logger.info("Creating Spanner graph schema (%d statements)...", len(create_ddl))
    database.update_ddl(create_ddl).result()
    logger.info("Spanner graph schema created")


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


_NODE_COLUMNS = [
    "id", "node_type", "name",
    "call_category", "call_outcome", "call_timestamp",
    "customer_id", "customer_type", "overall_sentiment",
    "agent_id", "agent_role",
    "issue_type", "severity", "issue_description",
    "product_name", "product_type",
    "service_name", "service_category",
    "solution_type", "resolution_status",
    "feedback_type", "feedback_sentiment",
    "document_ids", "description",
]

_EDGE_COLUMNS = [
    "id", "target_node_id", "edge_id",
    "relationship_type", "description", "weight", "document_ids",
]


def bulk_write_nodes(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    """Write graph nodes to the Nodes table."""
    _bulk_upsert(cfg, "Nodes", _NODE_COLUMNS, rows)


def bulk_write_edges(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    """Write graph edges to the Relationships table (interleaved in Nodes)."""
    _bulk_upsert(cfg, "Relationships", _EDGE_COLUMNS, rows)


def _bulk_upsert(
    cfg: GraphRAGConfig,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    client = get_instance(cfg)
    instance = client.instance(cfg.spanner.instance_id)
    database = instance.database(cfg.spanner.database_id)

    values = [[row.get(col) for col in columns] for row in rows]

    for chunk in _chunk_list(values, 500):
        with database.batch() as batch:
            batch.insert_or_update(table=table, columns=columns, values=chunk)

    logger.info("Upserted %d rows into Spanner %s", len(rows), table)
