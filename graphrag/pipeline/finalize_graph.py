"""Step 3: Finalize the entity/relationship graph -- degree, IDs, optional GraphML."""

from __future__ import annotations

import logging
from typing import Any

import igraph as ig
from google.cloud import bigquery

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

ENTITIES_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("type", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("human_readable_id", "INTEGER"),
    bigquery.SchemaField("degree", "INTEGER"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
]

RELATIONSHIPS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("target", "STRING"),
    bigquery.SchemaField("source_entity_id", "STRING"),
    bigquery.SchemaField("target_entity_id", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("weight", "FLOAT"),
    bigquery.SchemaField("human_readable_id", "INTEGER"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
]


def run(cfg: GraphRAGConfig, *, graphml_path: str | None = None) -> None:
    logger.info("Step 3: Finalizing graph")

    entities_raw = bq.read_table_all(cfg, "entities_raw")
    rels_raw = bq.read_table_all(cfg, "relationships_raw")

    # Build igraph
    title_to_entity: dict[str, dict[str, Any]] = {}
    for e in entities_raw:
        title_to_entity[e["title"]] = e

    g = ig.Graph(directed=False)
    titles = list(title_to_entity.keys())
    g.add_vertices(len(titles))
    g.vs["title"] = titles
    g.vs["entity_id"] = [title_to_entity[t]["id"] for t in titles]

    title_to_idx = {t: i for i, t in enumerate(titles)}

    valid_rels: list[dict[str, Any]] = []
    for r in rels_raw:
        src_idx = title_to_idx.get(r["source"])
        tgt_idx = title_to_idx.get(r["target"])
        if src_idx is not None and tgt_idx is not None:
            valid_rels.append(r)

    if valid_rels:
        edges = [
            (title_to_idx[r["source"]], title_to_idx[r["target"]])
            for r in valid_rels
        ]
        weights = [r.get("weight", 1.0) for r in valid_rels]
        g.add_edges(edges)
        g.es["weight"] = weights

    # Compute degree
    degrees = g.degree()

    # Sort entities by degree descending, assign human_readable_id
    indexed_entities: list[tuple[int, dict[str, Any]]] = []
    for i, title in enumerate(titles):
        e = dict(title_to_entity[title])
        e["degree"] = degrees[i]
        indexed_entities.append((degrees[i], e))

    indexed_entities.sort(key=lambda x: x[0], reverse=True)

    entity_rows: list[dict[str, Any]] = []
    for hrid, (_, e) in enumerate(indexed_entities):
        e["human_readable_id"] = hrid
        entity_rows.append(e)

    # Sort relationships by weight descending, assign human_readable_id,
    # resolve entity IDs
    id_by_title = {e["title"]: e["id"] for e in entity_rows}

    rel_with_weight: list[tuple[float, dict[str, Any]]] = []
    for r in valid_rels:
        r_copy = dict(r)
        r_copy["source_entity_id"] = id_by_title.get(r["source"], "")
        r_copy["target_entity_id"] = id_by_title.get(r["target"], "")
        rel_with_weight.append((r_copy.get("weight", 1.0), r_copy))

    rel_with_weight.sort(key=lambda x: x[0], reverse=True)

    rel_rows: list[dict[str, Any]] = []
    for hrid, (_, r) in enumerate(rel_with_weight):
        r["human_readable_id"] = hrid
        rel_rows.append(r)

    bq.write_rows(cfg, "entities", entity_rows, ENTITIES_SCHEMA)
    bq.write_rows(cfg, "relationships", rel_rows, RELATIONSHIPS_SCHEMA)
    logger.info(
        "Finalized %d entities and %d relationships",
        len(entity_rows), len(rel_rows),
    )

    if graphml_path:
        g.vs["label"] = g.vs["title"]
        g.write_graphml(graphml_path)
        logger.info("Exported GraphML to %s", graphml_path)
