"""Exploratory analysis of raw nodes using Splink.

Reads raw_nodes from BigQuery and produces per-node-type missingness and
column-distribution charts (HTML) to guide entity-resolution tuning.

Usage:
    python exploratory_analysis.py [--config config.yaml] [--output-dir eda_output]

Based on: https://moj-analytical-services.github.io/splink/demos/tutorials/02_Exploratory_analysis.html
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from splink import DuckDBAPI
from splink.exploratory import completeness_chart, profile_columns

from graphrag.config import GraphRAGConfig
from graphrag.models import NodeType
from graphrag.pipeline.entity_resolution import _normalize_name, _TYPE_COLUMNS
from graphrag.storage import bigquery as bq

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ALL_RAW_COLUMNS = [
    "id", "document_id", "node_type", "name",
    "call_category", "call_outcome", "call_timestamp",
    "customer_id", "customer_type", "overall_sentiment",
    "agent_id", "agent_role",
    "issue_type", "severity", "issue_description",
    "product_name", "product_type",
    "service_name", "service_category",
    "solution_type", "resolution_status",
    "feedback_type", "feedback_sentiment",
    "description",
]


def _save_chart(chart, path: Path) -> None:
    """Save a Splink Altair/VegaLite chart to an HTML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html = chart.to_html() if hasattr(chart, "to_html") else str(chart)
    path.write_text(html, encoding="utf-8")
    logger.info("Saved chart -> %s", path)


def _load_raw_nodes(cfg: GraphRAGConfig) -> pd.DataFrame:
    """Read all raw_nodes from BigQuery into a DataFrame."""
    rows = bq.read_table_all(cfg, "raw_nodes")
    if not rows:
        raise SystemExit("No raw_nodes found in BigQuery. Run the extraction pipeline first.")
    df = pd.DataFrame(rows)
    logger.info("Loaded %d raw nodes (%d columns)", len(df), len(df.columns))
    return df


def _relevant_columns(node_type: str) -> list[str]:
    """Return the columns that are meaningful for a given node type."""
    base = ["name"]
    extra = _TYPE_COLUMNS.get(node_type, [])
    return base + extra


def analyse_all(df: pd.DataFrame, out: Path) -> None:
    """Run completeness and profile analysis across all nodes."""
    db_api = DuckDBAPI()

    logger.info("=== Overall analysis (%d nodes) ===", len(df))
    chart = completeness_chart(df, db_api=db_api)
    _save_chart(chart, out / "all_completeness.html")

    cols_to_profile = ["node_type", "name", "description"]
    chart = profile_columns(
        df, db_api=db_api, column_expressions=cols_to_profile, top_n=15, bottom_n=5,
    )
    _save_chart(chart, out / "all_profile.html")

    logger.info("--- Node type distribution ---")
    print(df["node_type"].value_counts().to_string())
    print()


def analyse_node_type(df_type: pd.DataFrame, node_type: str, out: Path) -> None:
    """Run completeness and profile analysis for a single node type."""
    db_api = DuckDBAPI()
    slug = node_type.lower()
    cols = _relevant_columns(node_type)
    df_subset = df_type[["id", "document_id"] + cols].copy()

    logger.info("--- %s: completeness ---", node_type)
    chart = completeness_chart(df_subset, db_api=db_api)
    _save_chart(chart, out / f"{slug}_completeness.html")

    logger.info("--- %s: column profiles ---", node_type)
    chart = profile_columns(
        df_subset, db_api=db_api, column_expressions=cols, top_n=15, bottom_n=5,
    )
    _save_chart(chart, out / f"{slug}_profile.html")

    logger.info("--- %s: normalized-name cardinality ---", node_type)
    names = df_type["name"].dropna().apply(_normalize_name)
    total = len(names)
    unique = names.nunique()
    print(f"  {node_type}: {total} raw names -> {unique} unique normalized ({total - unique} potential duplicates)")
    print(f"  Top 10 most frequent names:")
    for name, count in names.value_counts().head(10).items():
        print(f"    {count:>5d}  {name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Splink exploratory analysis on raw nodes")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="eda_output", help="Directory for HTML charts")
    args = parser.parse_args()

    cfg = GraphRAGConfig.from_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_raw_nodes(cfg)

    analyse_all(df, out)

    for nt in NodeType:
        df_type = df[df["node_type"] == nt.value]
        if df_type.empty:
            logger.info("Skipping %s (no nodes)", nt.value)
            continue
        logger.info("=== %s: %d nodes ===", nt.value, len(df_type))
        analyse_node_type(df_type, nt.value, out)

    logger.info("All charts saved to %s/", out)


if __name__ == "__main__":
    main()
