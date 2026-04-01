from __future__ import annotations

import logging
from typing import Any

from google.cloud import spanner

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS Documents (
        id STRING(64) NOT NULL,
        title STRING(MAX),
        raw_content STRING(MAX),
        entity_ids ARRAY<STRING(36)>,
        relationship_ids ARRAY<STRING(36)>,
        content_embedding ARRAY<FLOAT32>(vector_length=>768)
    ) PRIMARY KEY (id)
    """,
    """
    CREATE TABLE IF NOT EXISTS Entities (
        id STRING(36) NOT NULL,
        title STRING(MAX),
        type STRING(256),
        description STRING(MAX),
        human_readable_id INT64,
        degree INT64,
        document_ids ARRAY<STRING(64)>,
        community_ids ARRAY<STRING(36)>,
        description_embedding ARRAY<FLOAT32>(vector_length=>768)
    ) PRIMARY KEY (id)
    """,
    """
    CREATE TABLE IF NOT EXISTS Relationships (
        id STRING(36) NOT NULL,
        source_entity_id STRING(36) NOT NULL,
        target_entity_id STRING(36) NOT NULL,
        description STRING(MAX),
        weight FLOAT64,
        human_readable_id INT64,
        document_ids ARRAY<STRING(64)>
    ) PRIMARY KEY (id)
    """,
    """
    CREATE TABLE IF NOT EXISTS Communities (
        id STRING(36) NOT NULL,
        level INT64,
        title STRING(MAX),
        summary STRING(MAX),
        full_content STRING(MAX),
        rating FLOAT64,
        rating_explanation STRING(MAX),
        entity_ids ARRAY<STRING(36)>,
        relationship_ids ARRAY<STRING(36)>,
        document_ids ARRAY<STRING(64)>,
        parent_community_id STRING(36),
        child_community_ids ARRAY<STRING(36)>,
        full_content_embedding ARRAY<FLOAT32>(vector_length=>768)
    ) PRIMARY KEY (id)
    """,
]

VECTOR_INDEX_STATEMENTS = [
    """
    CREATE VECTOR INDEX IF NOT EXISTS EntitiesEmbIdx
        ON Entities(description_embedding)
        OPTIONS (distance_type = 'COSINE', tree_depth = 3, num_leaves = 1000)
    """,
    """
    CREATE VECTOR INDEX IF NOT EXISTS CommunitiesEmbIdx
        ON Communities(full_content_embedding)
        OPTIONS (distance_type = 'COSINE', tree_depth = 3, num_leaves = 1000)
    """,
    """
    CREATE VECTOR INDEX IF NOT EXISTS DocumentsEmbIdx
        ON Documents(content_embedding)
        OPTIONS (distance_type = 'COSINE', tree_depth = 3, num_leaves = 1000)
    """,
]

PROPERTY_GRAPH_DDL = """
    CREATE OR REPLACE PROPERTY GRAPH KnowledgeGraph
        NODE TABLES (Entities, Documents, Communities)
        EDGE TABLES (
            Relationships
                SOURCE KEY (source_entity_id) REFERENCES Entities(id)
                DESTINATION KEY (target_entity_id) REFERENCES Entities(id)
        )
"""


def get_instance(cfg: GraphRAGConfig) -> spanner.Client:
    return spanner.Client(project=cfg.gcp.project_id)


def create_schema(cfg: GraphRAGConfig) -> None:
    """Create Spanner tables, vector indexes, and property graph."""
    client = get_instance(cfg)
    instance = client.instance(cfg.spanner.instance_id)
    database = instance.database(cfg.spanner.database_id)

    all_ddl = DDL_STATEMENTS + VECTOR_INDEX_STATEMENTS + [PROPERTY_GRAPH_DDL]
    operation = database.update_ddl(all_ddl)
    logger.info("Applying Spanner DDL (%d statements)...", len(all_ddl))
    operation.result()
    logger.info("Spanner schema created/updated")


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def bulk_write_documents(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    _bulk_upsert(
        cfg,
        "Documents",
        columns=["id", "title", "raw_content", "entity_ids", "relationship_ids", "content_embedding"],
        rows=rows,
    )


def bulk_write_entities(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    _bulk_upsert(
        cfg,
        "Entities",
        columns=["id", "title", "type", "description", "human_readable_id", "degree",
                 "document_ids", "community_ids", "description_embedding"],
        rows=rows,
    )


def bulk_write_relationships(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    _bulk_upsert(
        cfg,
        "Relationships",
        columns=["id", "source_entity_id", "target_entity_id", "description",
                 "weight", "human_readable_id", "document_ids"],
        rows=rows,
    )


def bulk_write_communities(cfg: GraphRAGConfig, rows: list[dict[str, Any]]) -> None:
    _bulk_upsert(
        cfg,
        "Communities",
        columns=["id", "level", "title", "summary", "full_content", "rating",
                 "rating_explanation", "entity_ids", "relationship_ids",
                 "document_ids", "parent_community_id", "child_community_ids",
                 "full_content_embedding"],
        rows=rows,
    )


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
