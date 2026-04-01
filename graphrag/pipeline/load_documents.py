"""Step 1: Load input documents from BigQuery."""

from __future__ import annotations

import hashlib
import logging

from google.cloud import bigquery

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

DOCUMENTS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("raw_content", "STRING"),
]


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 1: Loading documents from %s", cfg.source_table_fqn())
    bq.ensure_dataset(cfg)

    client = bq.get_client(cfg)
    query = f"SELECT * FROM `{cfg.source_table_fqn()}`"
    job = client.query(query)

    seen: set[str] = set()
    all_docs: list[dict] = []

    for row in job.result(page_size=cfg.pipeline.batch_size):
        row_dict = dict(row)
        text = row_dict.get("full_conversation") or ""
        title = row_dict.get("data_id") or ""
        doc_id = _doc_id(text)

        if doc_id in seen:
            continue
        seen.add(doc_id)

        all_docs.append({
            "id": doc_id,
            "title": str(title),
            "raw_content": text,
        })

    if not all_docs:
        logger.warning("No documents found in source table")
        return

    bq.write_rows(cfg, "documents", all_docs, DOCUMENTS_SCHEMA)
    logger.info("Step 1 complete: loaded %d unique documents", len(all_docs))
