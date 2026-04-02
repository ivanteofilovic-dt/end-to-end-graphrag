"""CLI entrypoint for the Graph RAG indexing pipeline."""

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
    write_spanner,
)

STEPS = {
    1: ("extract_graph", extract_graph.run),
    2: ("entity_resolution", entity_resolution.run),
    3: ("write_to_spanner", write_spanner.run),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telecom Knowledge Graph Indexing Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run only this step (1-3)",
    )
    parser.add_argument(
        "--from-step", type=int, default=None, dest="from_step",
        help="Run from this step onwards (1-3)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, dest="max_rows",
        help="Limit how many documents step 1 processes (e.g. 100 for a test run)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("graphrag")

    cfg = GraphRAGConfig.from_yaml(args.config)
    logger.info("Loaded config from %s (project=%s)", args.config, cfg.gcp.project_id)

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


if __name__ == "__main__":
    main()
