"""CLI entrypoint for the Graph RAG indexing pipeline and query interface."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq
from graphrag.pipeline import (
    extract_graph,
    entity_resolution,
    leiden_clustering,
    community_reports,
    write_spanner,
)

STEPS = {
    1: ("extract_graph", extract_graph.run),
    2: ("entity_resolution", entity_resolution.run),
    3: ("leiden_clustering", leiden_clustering.run),
    4: ("community_reports", community_reports.run),
    5: ("write_to_spanner", write_spanner.run),
}


def _run_index(args, cfg: GraphRAGConfig, logger: logging.Logger) -> None:
    """Run the indexing pipeline (steps 1-3)."""
    if args.step is not None:
        steps_to_run = [args.step]
    elif args.from_step is not None:
        steps_to_run = [s for s in sorted(STEPS) if s >= args.from_step]
    else:
        steps_to_run = sorted(STEPS)

    for step_num in steps_to_run:
        if step_num not in STEPS:
            logger.error("Unknown step %d. Valid steps: %s", step_num, list(STEPS.keys()))
            sys.exit(1)

    bq.ensure_dataset(cfg)

    total_start = time.time()

    for step_num in steps_to_run:
        name, fn = STEPS[step_num]
        logger.info("=" * 60)
        logger.info("Starting step %d: %s", step_num, name)
        logger.info("=" * 60)
        step_start = time.time()

        if step_num == 1 and args.max_rows:
            fn(cfg, max_rows=args.max_rows)
        else:
            fn(cfg)

        elapsed = time.time() - step_start
        logger.info("Step %d (%s) completed in %.1fs", step_num, name, elapsed)

    total_elapsed = time.time() - total_start
    logger.info("Pipeline finished in %.1fs", total_elapsed)


def _run_query(args, cfg: GraphRAGConfig) -> None:
    """Run a single question or start an interactive query session."""
    from graphrag.query import ask, interactive

    if args.question:
        answer = ask(cfg, args.question)
        print(answer)
    else:
        interactive(cfg)


def _run_batch(args, cfg: GraphRAGConfig, logger: logging.Logger) -> None:
    """Manage Gemini batch jobs (list / status / poll)."""
    from graphrag.batch import client as batch_client

    if args.batch_command == "list":
        jobs = batch_client.list_batch_jobs(cfg, limit=args.limit)
        if not jobs:
            print("No batch jobs found.")
            return
        for job in jobs:
            print(f"  {job.name}  state={job.state}")

    elif args.batch_command == "status":
        job = batch_client.get_batch_job(cfg, args.job_name)
        print(f"Job:   {job.name}")
        print(f"State: {job.state}")
        print(f"Model: {job.model}")
        if hasattr(job, "create_time") and job.create_time:
            print(f"Created: {job.create_time}")
        if hasattr(job, "update_time") and job.update_time:
            print(f"Updated: {job.update_time}")

    elif args.batch_command == "poll":
        logger.info("Resuming polling for job %s", args.job_name)
        batch_client.poll_until_done(cfg, args.job_name)
        logger.info("Batch job completed: %s", args.job_name)

    else:
        print("Usage: graphrag batch {list,status,poll}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telecom Knowledge Graph — Indexing Pipeline & Query Interface"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- index subcommand (default behaviour) ---
    index_parser = subparsers.add_parser(
        "index", help="Run the indexing pipeline (extract → resolve → cluster → report → write)",
    )
    index_parser.add_argument(
        "--step", type=int, default=None,
        help="Run only this step (1-5)",
    )
    index_parser.add_argument(
        "--from-step", type=int, default=None, dest="from_step",
        help="Run from this step onwards (1-5)",
    )
    index_parser.add_argument(
        "--max-rows", type=int, default=None, dest="max_rows",
        help="Limit how many documents step 1 processes (e.g. 100 for a test run)",
    )

    # --- query subcommand ---
    query_parser = subparsers.add_parser(
        "query", help="Ask questions over the Spanner knowledge graph",
    )
    query_parser.add_argument(
        "question", nargs="?", default=None,
        help="Question to ask (omit for interactive mode)",
    )

    # --- batch subcommand ---
    batch_parser = subparsers.add_parser(
        "batch", help="Manage Gemini batch prediction jobs",
    )
    batch_sub = batch_parser.add_subparsers(dest="batch_command")

    list_parser = batch_sub.add_parser("list", help="List recent batch jobs")
    list_parser.add_argument(
        "--limit", type=int, default=20,
        help="Max number of jobs to show (default: 20)",
    )

    status_parser = batch_sub.add_parser(
        "status", help="Show details of a batch job",
    )
    status_parser.add_argument("job_name", help="Batch job resource name")

    poll_parser = batch_sub.add_parser(
        "poll", help="Resume polling a running batch job until completion",
    )
    poll_parser.add_argument("job_name", help="Batch job resource name")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("graphrag")

    cfg = GraphRAGConfig.from_yaml(args.config)
    logger.info("Loaded config from %s (project=%s)", args.config, cfg.gcp.project_id)

    if args.command == "query":
        _run_query(args, cfg)
    elif args.command == "index":
        _run_index(args, cfg, logger)
    elif args.command == "batch":
        _run_batch(args, cfg, logger)
    else:
        # Backwards compat: no subcommand → run full indexing pipeline
        args.step = None
        args.from_step = None
        args.max_rows = None
        _run_index(args, cfg, logger)


if __name__ == "__main__":
    main()
