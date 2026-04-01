"""Step 1: Load input documents from BigQuery source table."""

from __future__ import annotations

import logging

from google.cloud import bigquery

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq

logger = logging.getLogger(__name__)

DOCUMENTS_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("raw_content", "STRING"),
    bigquery.SchemaField("conversation_date", "STRING"),
]


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 1: Loading documents from %s", cfg.source_table_fqn())
    bq.ensure_dataset(cfg)

    client = bq.get_client(cfg)
    query = f"SELECT data_id, full_conversation, conversation_date FROM `{cfg.source_table_fqn()}`"
    job = client.query(query)

    seen: set[str] = set()
    all_docs: list[dict] = []

    for row in job.result(page_size=cfg.pipeline.batch_size):
        row_dict = dict(row)
        data_id = str(row_dict.get("data_id") or "")
        text = row_dict.get("full_conversation") or ""
        conv_date = str(row_dict.get("conversation_date") or "")

        if not data_id or data_id in seen:
            continue
        seen.add(data_id)

        all_docs.append({
            "id": data_id,
            "raw_content": text,
            "conversation_date": conv_date,
        })

    if not all_docs:
        logger.warning("No documents found in source table")
        return

    bq.write_rows(cfg, "documents", all_docs, DOCUMENTS_SCHEMA)
    logger.info("Step 1 complete: loaded %d unique documents", len(all_docs))
