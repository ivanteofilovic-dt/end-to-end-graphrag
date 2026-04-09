"""Step 3: Leiden community detection on the merged knowledge graph.

Uses graspologic's hierarchical Leiden algorithm to produce a multi-level
community hierarchy.  Each unique community gets an entry in BigQuery's
``community_info`` table, and every (node, community, level) assignment is
stored in ``community_assignments``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
from google.cloud import bigquery
from graspologic.partition import hierarchical_leiden

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

COMMUNITY_ASSIGNMENTS_SCHEMA = [
    bigquery.SchemaField("node_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("community_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "INTEGER", mode="REQUIRED"),
]

COMMUNITY_INFO_SCHEMA = [
    bigquery.SchemaField("community_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("parent_community_id", "STRING"),
    bigquery.SchemaField("size", "INTEGER"),
]


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 3: Leiden community detection")

    nodes = bq.read_table_all(cfg, "merged_nodes")
    relationships = bq.read_table_all(cfg, "merged_relationships")

    if not nodes:
        logger.warning("No merged nodes found; skipping clustering")
        return

    graph = _build_graph(nodes, relationships)
    logger.info(
        "Built graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph; skipping clustering")
        return

    results = hierarchical_leiden(
        graph,
        max_cluster_size=cfg.community.max_cluster_size,
        random_seed=cfg.community.seed,
    )

    assignments, communities = _process_results(results)

    bq.write_rows(cfg, "community_assignments", assignments, COMMUNITY_ASSIGNMENTS_SCHEMA)
    bq.write_rows(cfg, "community_info", communities, COMMUNITY_INFO_SCHEMA)

    n_levels = len({c["level"] for c in communities})
    logger.info(
        "Detected %d communities across %d levels (from %d nodes)",
        len(communities),
        n_levels,
        graph.number_of_nodes(),
    )
    logger.info("Step 3 complete")


# ── Graph construction ───────────────────────────────────────────────────


def _build_graph(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> nx.Graph:
    """Build an undirected weighted NetworkX graph from merged nodes/edges."""
    graph = nx.Graph()

    node_ids: set[str] = set()
    for n in nodes:
        nid = n["id"]
        graph.add_node(nid)
        node_ids.add(nid)

    for r in relationships:
        src = r.get("source_node_id", "")
        tgt = r.get("target_node_id", "")
        if src in node_ids and tgt in node_ids and src != tgt:
            weight = float(r.get("weight", 1.0))
            if graph.has_edge(src, tgt):
                graph[src][tgt]["weight"] += weight
            else:
                graph.add_edge(src, tgt, weight=weight)

    return graph


# ── Result processing ────────────────────────────────────────────────────


def _process_results(
    results: list,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert hierarchical_leiden output into BQ-ready assignment and info rows.

    Each result entry has attributes: node, cluster, parent_cluster, level,
    is_final_cluster.  Cluster IDs are globally unique across levels.
    """
    cluster_members: dict[int, list[str]] = defaultdict(list)
    cluster_level: dict[int, int] = {}
    parent_map: dict[int, int | None] = {}

    for r in results:
        cluster_members[r.cluster].append(r.node)
        cluster_level[r.cluster] = r.level
        if r.cluster not in parent_map:
            parent_map[r.cluster] = r.parent_cluster

    assignment_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []

    for cluster_id, members in cluster_members.items():
        level = cluster_level[cluster_id]
        cid = str(cluster_id)

        for node_id in members:
            assignment_rows.append({
                "node_id": node_id,
                "community_id": cid,
                "level": level,
            })

        parent = parent_map.get(cluster_id)
        community_rows.append({
            "community_id": cid,
            "level": level,
            "parent_community_id": str(parent) if parent is not None else None,
            "size": len(members),
        })

    return assignment_rows, community_rows
