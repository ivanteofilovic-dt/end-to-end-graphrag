from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from google.cloud import bigquery

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)


def get_client(cfg: GraphRAGConfig) -> bigquery.Client:
    return bigquery.Client(project=cfg.gcp.project_id, location=cfg.gcp.location)


def ensure_dataset(cfg: GraphRAGConfig) -> None:
    client = get_client(cfg)
    dataset_ref = bigquery.DatasetReference(
        cfg.gcp.project_id, cfg.bigquery.intermediate_dataset
    )
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = cfg.gcp.location
    client.create_dataset(dataset, exists_ok=True)
    logger.info("Ensured dataset %s exists", cfg.bigquery.intermediate_dataset)


def read_table(
    cfg: GraphRAGConfig,
    table_name: str,
    *,
    columns: list[str] | None = None,
    where: str | None = None,
    batch_size: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of rows from a BigQuery table.

    Returns an empty iterator if the table does not exist.
    """
    if not table_exists(cfg, table_name):
        logger.warning("Table %s does not exist yet; returning empty result", table_name)
        return

    client = get_client(cfg)
    fqn = cfg.table_fqn(table_name)
    col_expr = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_expr} FROM `{fqn}`"
    if where:
        query += f" WHERE {where}"

    page_size = batch_size or cfg.pipeline.batch_size
    job = client.query(query)
    result = job.result(page_size=page_size)

    batch: list[dict[str, Any]] = []
    for row in result:
        batch.append(dict(row))
        if len(batch) >= page_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_table_all(
    cfg: GraphRAGConfig,
    table_name: str,
    *,
    columns: list[str] | None = None,
    where: str | None = None,
) -> list[dict[str, Any]]:
    """Read all rows from a BigQuery table."""
    rows: list[dict[str, Any]] = []
    for batch in read_table(cfg, table_name, columns=columns, where=where):
        rows.extend(batch)
    return rows


def write_rows(
    cfg: GraphRAGConfig,
    table_name: str,
    rows: list[dict[str, Any]],
    schema: list[bigquery.SchemaField],
    *,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    """Write rows to a BigQuery table, creating it if necessary."""
    client = get_client(cfg)
    fqn = cfg.table_fqn(table_name)
    table = bigquery.Table(fqn, schema=schema)
    client.create_table(table, exists_ok=True)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition,
    )
    job = client.load_table_from_json(rows, fqn, job_config=job_config)
    job.result()
    logger.info("Wrote %d rows to %s", len(rows), fqn)


def write_rows_append(
    cfg: GraphRAGConfig,
    table_name: str,
    rows: list[dict[str, Any]],
    schema: list[bigquery.SchemaField],
) -> None:
    write_rows(
        cfg, table_name, rows, schema, write_disposition="WRITE_APPEND"
    )


def run_query(cfg: GraphRAGConfig, query: str) -> list[dict[str, Any]]:
    client = get_client(cfg)
    job = client.query(query)
    return [dict(row) for row in job.result()]


def table_exists(cfg: GraphRAGConfig, table_name: str) -> bool:
    client = get_client(cfg)
    try:
        client.get_table(cfg.table_fqn(table_name))
        return True
    except Exception:
        return False


def row_count(cfg: GraphRAGConfig, table_name: str) -> int:
    result = run_query(cfg, f"SELECT COUNT(*) as cnt FROM `{cfg.table_fqn(table_name)}`")
    return result[0]["cnt"]


def _serialize_for_bq(value: Any) -> Any:
    """Recursively serialize values for BigQuery JSON compatibility."""
    if isinstance(value, dict):
        return {k: _serialize_for_bq(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_bq(v) for v in value]
    return value


def write_batch_request_table(
    cfg: GraphRAGConfig,
    table_name: str,
    rows: list[dict[str, Any]],
) -> str:
    """Write a Gemini Batch API request table.

    Each row must have a ``request`` key (dict matching GenerateContentRequest)
    and any number of pass-through columns (STRING typed).

    Returns the ``bq://`` URI.
    """
    schema_fields: list[bigquery.SchemaField] = []
    pass_through_keys: list[str] = []
    for key in rows[0]:
        if key == "request":
            schema_fields.append(bigquery.SchemaField("request", "JSON"))
        else:
            schema_fields.append(bigquery.SchemaField(key, "STRING"))
            pass_through_keys.append(key)

    serialized = []
    for row in rows:
        out: dict[str, Any] = {}
        for key in pass_through_keys:
            out[key] = str(row[key])
        out["request"] = row["request"]
        serialized.append(out)

    write_rows(cfg, table_name, serialized, schema_fields)
    return f"bq://{cfg.table_fqn(table_name)}"
