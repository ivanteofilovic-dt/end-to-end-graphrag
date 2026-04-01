"""Step 3: Entity resolution via Splink 4 (DuckDB backend).

For each entity type (except Call, which is 1:1 with transcripts) we run
probabilistic deduplication using Splink, then merge clusters into canonical
nodes and remap all relationships accordingly.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any

import pandas as pd
import splink.comparison_library as cl
from google.cloud import bigquery
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from graphrag.config import GraphRAGConfig
from graphrag.models import NodeType
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

MERGED_NODES_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("node_type", "STRING"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
    # Call
    bigquery.SchemaField("call_category", "STRING"),
    bigquery.SchemaField("call_outcome", "STRING"),
    bigquery.SchemaField("call_timestamp", "STRING"),
    # Customer
    bigquery.SchemaField("customer_id", "STRING"),
    bigquery.SchemaField("customer_type", "STRING"),
    bigquery.SchemaField("overall_sentiment", "STRING"),
    # Agent
    bigquery.SchemaField("agent_id", "STRING"),
    bigquery.SchemaField("agent_role", "STRING"),
    # Problem
    bigquery.SchemaField("issue_type", "STRING"),
    bigquery.SchemaField("severity", "STRING"),
    bigquery.SchemaField("issue_description", "STRING"),
    # Product
    bigquery.SchemaField("product_name", "STRING"),
    bigquery.SchemaField("product_type", "STRING"),
    # Service
    bigquery.SchemaField("service_name", "STRING"),
    bigquery.SchemaField("service_category", "STRING"),
    # Solution
    bigquery.SchemaField("solution_type", "STRING"),
    bigquery.SchemaField("resolution_status", "STRING"),
    # Feedback
    bigquery.SchemaField("feedback_type", "STRING"),
    bigquery.SchemaField("feedback_sentiment", "STRING"),
    # Generic
    bigquery.SchemaField("description", "STRING"),
]

MERGED_RELATIONSHIPS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("relationship_type", "STRING"),
    bigquery.SchemaField("source_node_id", "STRING"),
    bigquery.SchemaField("target_node_id", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("weight", "FLOAT"),
    bigquery.SchemaField("document_ids", "STRING", mode="REPEATED"),
]

# Splink comparison configs keyed by NodeType value.
# Each entry is (comparisons, blocking_rules).
_SPLINK_CONFIGS: dict[str, dict] = {
    NodeType.CUSTOMER: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("customer_type"),
        ],
        "blocking": [
            block_on("customer_type"),
            block_on("name"),
        ],
    },
    NodeType.AGENT: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("agent_role"),
        ],
        "blocking": [
            block_on("agent_role"),
            block_on("name"),
        ],
    },
    NodeType.PROBLEM: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.JaroWinklerAtThresholds("issue_type", [0.9, 0.7]),
        ],
        "blocking": [
            block_on("issue_type"),
            block_on("name"),
        ],
    },
    NodeType.PRODUCT: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("product_type"),
        ],
        "blocking": [
            block_on("product_type"),
            block_on("name"),
        ],
    },
    NodeType.SERVICE: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("service_category"),
        ],
        "blocking": [
            block_on("service_category"),
            block_on("name"),
        ],
    },
    NodeType.SOLUTION: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("resolution_status"),
        ],
        "blocking": [
            block_on("resolution_status"),
            block_on("name"),
        ],
    },
    NodeType.FEEDBACK: {
        "comparisons": [
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("feedback_type"),
        ],
        "blocking": [
            block_on("feedback_type"),
            block_on("name"),
        ],
    },
}

# Columns to carry through per node type (besides id, name, document_id, node_type).
_TYPE_COLUMNS: dict[str, list[str]] = {
    NodeType.CALL: [
        "call_category", "call_outcome", "call_timestamp", "description",
    ],
    NodeType.CUSTOMER: [
        "customer_id", "customer_type", "overall_sentiment", "description",
    ],
    NodeType.AGENT: [
        "agent_id", "agent_role", "description",
    ],
    NodeType.PROBLEM: [
        "issue_type", "severity", "issue_description",
    ],
    NodeType.PRODUCT: [
        "product_name", "product_type", "description",
    ],
    NodeType.SERVICE: [
        "service_name", "service_category", "description",
    ],
    NodeType.SOLUTION: [
        "solution_type", "resolution_status", "description",
    ],
    NodeType.FEEDBACK: [
        "feedback_type", "feedback_sentiment", "description",
    ],
}


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 3: Entity resolution with Splink")

    raw_nodes = bq.read_table_all(cfg, "raw_nodes")
    raw_rels = bq.read_table_all(cfg, "raw_relationships")

    if not raw_nodes:
        logger.warning("No raw nodes to resolve")
        return

    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for n in raw_nodes:
        nodes_by_type[n["node_type"]].append(n)

    # old_id -> canonical_id mapping
    id_remap: dict[str, str] = {}
    merged_nodes: list[dict[str, Any]] = []

    for node_type in NodeType:
        type_nodes = nodes_by_type.get(node_type.value, [])
        if not type_nodes:
            continue

        if node_type == NodeType.CALL:
            canonical = _pass_through_calls(type_nodes)
        else:
            canonical = _resolve_type(cfg, node_type.value, type_nodes)

        for canon in canonical:
            for old_id in canon.pop("_source_ids", [canon["id"]]):
                id_remap[old_id] = canon["id"]
            merged_nodes.append(canon)

    logger.info("Resolved to %d merged nodes (from %d raw)", len(merged_nodes), len(raw_nodes))

    # Build name->canonical_id lookup for relationship remapping
    name_to_canonical = _build_name_lookup(raw_nodes, id_remap)

    merged_rels = _remap_and_deduplicate_relationships(raw_rels, name_to_canonical)
    logger.info("Resolved to %d merged relationships (from %d raw)", len(merged_rels), len(raw_rels))

    bq.write_rows(cfg, "merged_nodes", merged_nodes, MERGED_NODES_SCHEMA)
    bq.write_rows(cfg, "merged_relationships", merged_rels, MERGED_RELATIONSHIPS_SCHEMA)

    logger.info("Step 3 complete")


def _pass_through_calls(nodes: list[dict]) -> list[dict[str, Any]]:
    """Call nodes are 1:1 with transcripts -- no resolution needed."""
    results = []
    for n in nodes:
        canon = _make_canonical(n, [n])
        canon["_source_ids"] = [n["id"]]
        results.append(canon)
    return results


def _resolve_type(
    cfg: GraphRAGConfig,
    node_type: str,
    nodes: list[dict],
) -> list[dict[str, Any]]:
    """Run Splink deduplication for one entity type."""
    splink_cfg = _SPLINK_CONFIGS.get(node_type)
    if splink_cfg is None or len(nodes) < 2:
        return [_make_canonical(n, [n]) | {"_source_ids": [n["id"]]} for n in nodes]

    extra_cols = _TYPE_COLUMNS.get(node_type, [])
    records = []
    for n in nodes:
        rec: dict[str, Any] = {
            "unique_id": n["id"],
            "name": n.get("name") or "",
            "document_id": n.get("document_id") or "",
        }
        for col in extra_cols:
            rec[col] = n.get(col) or ""
        records.append(rec)

    df = pd.DataFrame(records)

    # Replace empty strings with None so Splink treats them as missing
    df = df.replace("", None)

    db_api = DuckDBAPI()
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=splink_cfg["comparisons"],
        blocking_rules_to_generate_predictions=splink_cfg["blocking"],
    )

    linker = Linker(df, settings, db_api)

    try:
        linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)

        for rule in splink_cfg["blocking"]:
            try:
                linker.training.estimate_parameters_using_expectation_maximisation(rule)
            except Exception:
                logger.debug("EM training failed for rule %s on %s, skipping", rule, node_type)
                continue
    except Exception:
        logger.warning(
            "Splink training failed for %s (%d nodes), falling back to name-based merge",
            node_type, len(nodes),
        )
        return _fallback_name_merge(node_type, nodes)

    try:
        predictions = linker.inference.predict(
            threshold_match_weight=cfg.splink.match_weight_threshold
        )
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions,
            threshold_match_probability=cfg.splink.cluster_threshold,
        )
        cluster_df = clusters.as_pandas_dataframe()
    except Exception:
        logger.warning(
            "Splink prediction/clustering failed for %s, falling back to name-based merge",
            node_type,
        )
        return _fallback_name_merge(node_type, nodes)

    # Group nodes by cluster_id
    node_by_id = {n["id"]: n for n in nodes}
    clusters_map: dict[str, list[dict]] = defaultdict(list)
    for _, row in cluster_df.iterrows():
        uid = row["unique_id"]
        cid = str(row["cluster_id"])
        if uid in node_by_id:
            clusters_map[cid].append(node_by_id[uid])

    results: list[dict[str, Any]] = []
    for cluster_nodes in clusters_map.values():
        canonical = _make_canonical(cluster_nodes[0], cluster_nodes)
        canonical["_source_ids"] = [n["id"] for n in cluster_nodes]
        results.append(canonical)

    return results


def _fallback_name_merge(node_type: str, nodes: list[dict]) -> list[dict[str, Any]]:
    """Simple name-based merge when Splink cannot run."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_name[n.get("name", "").upper()].append(n)

    results = []
    for group in by_name.values():
        canonical = _make_canonical(group[0], group)
        canonical["_source_ids"] = [n["id"] for n in group]
        results.append(canonical)
    return results


def _make_canonical(representative: dict, members: list[dict]) -> dict[str, Any]:
    """Build a canonical node record from a cluster of raw nodes."""
    node_type = representative["node_type"]
    extra_cols = _TYPE_COLUMNS.get(node_type, [])
    doc_ids: set[str] = set()
    for m in members:
        did = m.get("document_id")
        if did:
            doc_ids.add(did)

    canon: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "node_type": node_type,
        "name": representative.get("name", ""),
        "document_ids": sorted(doc_ids),
    }

    for col in extra_cols:
        values = [m.get(col) for m in members if m.get(col)]
        canon[col] = values[0] if values else None

    return canon


def _build_name_lookup(
    raw_nodes: list[dict], id_remap: dict[str, str]
) -> dict[str, str]:
    """Map (document_id, name) pairs to canonical node IDs for relationship remapping.

    Also builds a global name->canonical_id fallback for cross-document edges.
    """
    doc_name_to_canonical: dict[tuple[str, str], str] = {}
    name_to_canonical: dict[str, str] = {}

    for n in raw_nodes:
        old_id = n["id"]
        canon_id = id_remap.get(old_id, old_id)
        name = (n.get("name") or "").upper()
        doc_id = n.get("document_id", "")

        if name:
            doc_name_to_canonical[(doc_id, name)] = canon_id
            name_to_canonical[name] = canon_id

    return name_to_canonical


def _remap_and_deduplicate_relationships(
    raw_rels: list[dict],
    name_to_canonical: dict[str, str],
) -> list[dict[str, Any]]:
    """Remap relationship endpoints to canonical node IDs and deduplicate."""
    EdgeKey = tuple[str, str, str]  # (source_id, target_id, rel_type)
    grouped: dict[EdgeKey, list[dict]] = defaultdict(list)

    for rel in raw_rels:
        src_name = (rel.get("source_name") or "").upper()
        tgt_name = (rel.get("target_name") or "").upper()

        src_id = name_to_canonical.get(src_name)
        tgt_id = name_to_canonical.get(tgt_name)

        if not src_id or not tgt_id:
            continue

        key: EdgeKey = (src_id, tgt_id, rel.get("relationship_type", ""))
        grouped[key].append(rel)

    merged: list[dict[str, Any]] = []
    for (src_id, tgt_id, rel_type), rels in grouped.items():
        doc_ids: set[str] = set()
        descriptions: list[str] = []
        total_weight = 0.0

        for r in rels:
            did = r.get("document_id")
            if did:
                doc_ids.add(did)
            desc = r.get("description")
            if desc:
                descriptions.append(desc)
            total_weight += r.get("weight", 1.0)

        merged.append({
            "id": str(uuid.uuid4()),
            "relationship_type": rel_type,
            "source_node_id": src_id,
            "target_node_id": tgt_id,
            "description": descriptions[0] if descriptions else None,
            "weight": total_weight / len(rels),
            "document_ids": sorted(doc_ids),
        })

    return merged
