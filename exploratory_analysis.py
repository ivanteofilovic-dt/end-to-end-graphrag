"""Exploratory analysis of raw and merged nodes using Splink.

Reads raw_nodes or merged_nodes from BigQuery and produces per-node-type
missingness and column-distribution charts (HTML) to guide entity-resolution
tuning or evaluate its results.

Usage:
    python exploratory_analysis.py --stage raw    [--config config.yaml] [--output-dir eda_output]
    python exploratory_analysis.py --stage merged [--config config.yaml] [--output-dir eda_output]
    python exploratory_analysis.py --stage both   [--config config.yaml] [--output-dir eda_output]

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


def _load_nodes(cfg: GraphRAGConfig, stage: str) -> pd.DataFrame:
    """Read all nodes for the given stage from BigQuery into a DataFrame."""
    table = "raw_nodes" if stage == "raw" else "merged_nodes"
    rows = bq.read_table_all(cfg, table)
    if not rows:
        raise SystemExit(f"No {table} found in BigQuery. Run the pipeline first.")
    df = pd.DataFrame(rows)
    logger.info("Loaded %d %s nodes (%d columns)", len(df), stage, len(df.columns))
    return df


def _id_columns(stage: str) -> list[str]:
    """Return the ID columns present in raw vs merged schemas."""
    if stage == "raw":
        return ["id", "document_id"]
    return ["id"]


def _relevant_columns(node_type: str) -> list[str]:
    """Return the columns that are meaningful for a given node type."""
    base = ["name"]
    extra = _TYPE_COLUMNS.get(node_type, [])
    return base + extra


def analyse_all(df: pd.DataFrame, stage: str, out: Path) -> None:
    """Run completeness and profile analysis across all nodes."""
    db_api = DuckDBAPI()
    label = stage.capitalize()

    logger.info("=== %s — overall analysis (%d nodes) ===", label, len(df))
    chart = completeness_chart(df, db_api=db_api)
    _save_chart(chart, out / f"{stage}_all_completeness.html")

    cols_to_profile = ["node_type", "name", "description"]
    chart = profile_columns(
        df, db_api=db_api, column_expressions=cols_to_profile, top_n=15, bottom_n=5,
    )
    _save_chart(chart, out / f"{stage}_all_profile.html")

    logger.info("--- %s — node type distribution ---", label)
    print(df["node_type"].value_counts().to_string())
    print()


def analyse_node_type(
    df_type: pd.DataFrame, node_type: str, stage: str, out: Path,
) -> None:
    """Run completeness and profile analysis for a single node type."""
    db_api = DuckDBAPI()
    slug = node_type.lower()
    label = stage.capitalize()
    cols = _relevant_columns(node_type)
    id_cols = [c for c in _id_columns(stage) if c in df_type.columns]
    available_cols = [c for c in cols if c in df_type.columns]
    df_subset = df_type[id_cols + available_cols].copy()

    logger.info("--- %s %s: completeness ---", label, node_type)
    chart = completeness_chart(df_subset, db_api=db_api)
    _save_chart(chart, out / f"{stage}_{slug}_completeness.html")

    logger.info("--- %s %s: column profiles ---", label, node_type)
    chart = profile_columns(
        df_subset, db_api=db_api, column_expressions=available_cols, top_n=15, bottom_n=5,
    )
    _save_chart(chart, out / f"{stage}_{slug}_profile.html")

    logger.info("--- %s %s: name cardinality ---", label, node_type)
    names = df_type["name"].dropna().apply(_normalize_name)
    total = len(names)
    unique = names.nunique()
    if stage == "raw":
        print(f"  {node_type}: {total} raw names -> {unique} unique normalized ({total - unique} potential duplicates)")
    else:
        print(f"  {node_type}: {total} merged nodes, {unique} unique normalized names")
    print(f"  Top 10 most frequent names:")
    for name, count in names.value_counts().head(10).items():
        print(f"    {count:>5d}  {name}")
    print()


def compare_stages(df_raw: pd.DataFrame, df_merged: pd.DataFrame) -> None:
    """Print a side-by-side comparison of raw vs merged node counts."""
    print("\n" + "=" * 60)
    print("  Raw vs Merged — Entity Resolution Summary")
    print("=" * 60)
    print(f"  {'Node Type':<12} {'Raw':>8} {'Merged':>8} {'Reduction':>10}")
    print("  " + "-" * 42)

    for nt in NodeType:
        raw_count = len(df_raw[df_raw["node_type"] == nt.value])
        merged_count = len(df_merged[df_merged["node_type"] == nt.value])
        if raw_count == 0:
            continue
        pct = (1 - merged_count / raw_count) * 100 if raw_count else 0
        print(f"  {nt.value:<12} {raw_count:>8d} {merged_count:>8d} {pct:>9.1f}%")

    raw_total = len(df_raw)
    merged_total = len(df_merged)
    pct_total = (1 - merged_total / raw_total) * 100 if raw_total else 0
    print("  " + "-" * 42)
    print(f"  {'TOTAL':<12} {raw_total:>8d} {merged_total:>8d} {pct_total:>9.1f}%")
    print("=" * 60 + "\n")


def _run_stage(cfg: GraphRAGConfig, stage: str, out: Path) -> pd.DataFrame:
    """Load nodes for *stage* and run all analyses. Returns the DataFrame."""
    df = _load_nodes(cfg, stage)

    # merged_nodes has document_ids (repeated/list) — not useful for Splink profiling
    drop = [c for c in ("document_ids",) if c in df.columns]
    df_analysis = df.drop(columns=drop)

    analyse_all(df_analysis, stage, out)

    for nt in NodeType:
        df_type = df_analysis[df_analysis["node_type"] == nt.value]
        if df_type.empty:
            logger.info("Skipping %s (no %s nodes)", nt.value, stage)
            continue
        logger.info("=== %s %s: %d nodes ===", stage.capitalize(), nt.value, len(df_type))
        analyse_node_type(df_type, nt.value, stage, out)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Splink exploratory analysis on raw and/or merged nodes",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="eda_output", help="Directory for HTML charts")
    parser.add_argument(
        "--stage",
        choices=["raw", "merged", "both"],
        default="both",
        help="Which node table to analyse (default: both)",
    )
    args = parser.parse_args()

    cfg = GraphRAGConfig.from_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = None
    df_merged = None

    if args.stage in ("raw", "both"):
        df_raw = _run_stage(cfg, "raw", out)

    if args.stage in ("merged", "both"):
        df_merged = _run_stage(cfg, "merged", out)

    if df_raw is not None and df_merged is not None:
        compare_stages(df_raw, df_merged)

    logger.info("All charts saved to %s/", out)


if __name__ == "__main__":
    main()
