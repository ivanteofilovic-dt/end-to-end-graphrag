"""Step 4: Create communities via Hierarchical Leiden clustering."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any

import igraph as ig
from google.cloud import bigquery
from graspologic.partition import hierarchical_leiden

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

COMMUNITIES_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("level", "INTEGER"),
    bigquery.SchemaField("entity_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("relationship_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("parent_community_id", "STRING"),
    bigquery.SchemaField("child_community_ids", "STRING", mode="REPEATED"),
]


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 4: Creating communities")

    entities = bq.read_table_all(cfg, "entities")
    relationships = bq.read_table_all(cfg, "relationships")

    # Build igraph
    title_to_entity = {e["title"]: e for e in entities}
    titles = list(title_to_entity.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(titles))
    g.vs["name"] = titles
    g.vs["entity_id"] = [title_to_entity[t]["id"] for t in titles]

    rel_lookup: dict[int, list[dict[str, Any]]] = defaultdict(list)
    edge_list: list[tuple[int, int]] = []
    weights: list[float] = []

    for r in relationships:
        si = title_to_idx.get(r["source"])
        ti = title_to_idx.get(r["target"])
        if si is not None and ti is not None:
            edge_idx = len(edge_list)
            edge_list.append((si, ti))
            weights.append(r.get("weight", 1.0))
            rel_lookup[edge_idx] = r

    if edge_list:
        g.add_edges(edge_list)
        g.es["weight"] = weights

    if g.vcount() == 0:
        logger.warning("Graph has no vertices; skipping community detection")
        return

    # Run Hierarchical Leiden
    community_mapping = hierarchical_leiden(
        g,
        max_cluster_size=len(titles),
        resolution=cfg.community.resolution,
    )

    # community_mapping is a list of HierarchicalCluster items with
    # .node, .cluster, .parent_cluster, .level, .is_final_cluster
    level_communities: dict[int, dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    node_cluster_parent: dict[int, dict[int, int | None]] = defaultdict(dict)

    for item in community_mapping:
        level_communities[item.level][item.cluster].add(item.node)
        if item.parent_cluster is not None:
            node_cluster_parent[item.level][item.cluster] = item.parent_cluster

    max_level = min(max(level_communities.keys(), default=0), cfg.community.max_levels - 1)

    # Build community records
    # community_key = (level, cluster_id) -> Community UUID
    community_uuid: dict[tuple[int, int], str] = {}
    for level in sorted(level_communities.keys()):
        if level > max_level:
            break
        for cluster_id in level_communities[level]:
            community_uuid[(level, cluster_id)] = str(uuid.uuid4())

    community_rows: list[dict[str, Any]] = []
    for level in sorted(level_communities.keys()):
        if level > max_level:
            break
        for cluster_id, node_indices in level_communities[level].items():
            c_id = community_uuid[(level, cluster_id)]

            entity_ids = [g.vs[n]["entity_id"] for n in node_indices]

            # Relationships where both endpoints are in this community
            rel_ids: list[str] = []
            for edge in g.es:
                if edge.source in node_indices and edge.target in node_indices:
                    r = rel_lookup.get(edge.index)
                    if r and "id" in r:
                        rel_ids.append(r["id"])

            # Document IDs from member entities
            doc_ids_set: set[str] = set()
            for n in node_indices:
                entity = title_to_entity.get(g.vs[n]["name"], {})
                for did in entity.get("document_ids", []):
                    doc_ids_set.add(did)

            # Parent community
            parent_cluster = node_cluster_parent.get(level, {}).get(cluster_id)
            parent_id = None
            if parent_cluster is not None and level > 0:
                parent_key = (level - 1, parent_cluster)
                parent_id = community_uuid.get(parent_key)

            # Child communities (next level down that map to this cluster)
            child_ids: list[str] = []
            if level < max_level:
                next_level = level + 1
                for next_cluster_id, next_nodes in level_communities.get(next_level, {}).items():
                    next_parent = node_cluster_parent.get(next_level, {}).get(next_cluster_id)
                    if next_parent == cluster_id:
                        child_key = (next_level, next_cluster_id)
                        child_uuid = community_uuid.get(child_key)
                        if child_uuid:
                            child_ids.append(child_uuid)

            community_rows.append({
                "id": c_id,
                "level": level,
                "entity_ids": entity_ids,
                "relationship_ids": rel_ids,
                "document_ids": list(doc_ids_set),
                "parent_community_id": parent_id,
                "child_community_ids": child_ids,
            })

    bq.write_rows(cfg, "communities", community_rows, COMMUNITIES_SCHEMA)
    logger.info(
        "Created %d communities across %d levels",
        len(community_rows),
        max_level + 1 if community_rows else 0,
    )
