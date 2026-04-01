"""Step 6: Generate embeddings and sync to Spanner Graph."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions

from graphrag.config import GraphRAGConfig
from graphrag.storage import bigquery as bq
from graphrag.storage import spanner

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_EMBED = 10


def run(cfg: GraphRAGConfig) -> None:
    logger.info("Step 6: Generating embeddings and syncing to Spanner")
    asyncio.run(_run_async(cfg))


async def _run_async(cfg: GraphRAGConfig) -> None:
    # 1) Embed documents
    documents = bq.read_table_all(cfg, "documents")
    logger.info("Embedding %d documents", len(documents))
    doc_texts = [d.get("raw_content", "") for d in documents]
    doc_embeddings = await _embed_texts(cfg, doc_texts)
    for doc, emb in zip(documents, doc_embeddings):
        doc["content_embedding"] = emb

    # 2) Embed entities
    entities = bq.read_table_all(cfg, "entities")
    logger.info("Embedding %d entities", len(entities))
    entity_texts = [
        f"{e.get('title', '')}: {e.get('description', '')}" for e in entities
    ]
    entity_embeddings = await _embed_texts(cfg, entity_texts)
    for ent, emb in zip(entities, entity_embeddings):
        ent["description_embedding"] = emb

    # 3) Embed community reports
    communities = bq.read_table_all(cfg, "communities")
    communities_with_content = [
        c for c in communities if c.get("full_content")
    ]
    logger.info(
        "Embedding %d community reports (of %d total)",
        len(communities_with_content), len(communities),
    )
    if communities_with_content:
        comm_texts = [c["full_content"] for c in communities_with_content]
        comm_embeddings = await _embed_texts(cfg, comm_texts)
        comm_emb_by_id: dict[str, list[float]] = {}
        for c, emb in zip(communities_with_content, comm_embeddings):
            comm_emb_by_id[c["id"]] = emb
        for c in communities:
            c["full_content_embedding"] = comm_emb_by_id.get(c["id"])

    # 4) Sync to Spanner
    relationships = bq.read_table_all(cfg, "relationships")
    _sync_to_spanner(cfg, documents, entities, relationships, communities)

    logger.info("Step 6 complete")


async def _embed_texts(
    cfg: GraphRAGConfig, texts: list[str]
) -> list[list[float]]:
    """Embed a list of texts using Vertex AI, batching per API limits."""
    if not texts:
        return []

    client = genai.Client(
        vertexai=True,
        project=cfg.gcp.project_id,
        location=cfg.gcp.location,
        http_options=HttpOptions(api_version="v1"),
    )

    batch_size = cfg.embedding.batch_size
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EMBED)
    all_embeddings: list[list[float]] = [[] for _ in texts]

    batches: list[tuple[int, int]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batches.append((start, end))

    async def _embed_batch(start: int, end: int) -> None:
        async with semaphore:
            batch_texts = texts[start:end]
            sanitized = [t if t.strip() else " " for t in batch_texts]

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.embed_content(
                    model=cfg.embedding.model,
                    contents=sanitized,
                    config=EmbedContentConfig(
                        task_type=cfg.embedding.task_type,
                        output_dimensionality=cfg.embedding.dimensions,
                    ),
                ),
            )
            for i, embedding in enumerate(response.embeddings):
                all_embeddings[start + i] = list(embedding.values)

    tasks = [_embed_batch(s, e) for s, e in batches]
    await asyncio.gather(*tasks)

    return all_embeddings


def _sync_to_spanner(
    cfg: GraphRAGConfig,
    documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    communities: list[dict[str, Any]],
) -> None:
    """Create Spanner schema and bulk-write all data."""
    logger.info("Creating Spanner schema")
    spanner.create_schema(cfg)

    # Prepare document rows
    doc_rows = [
        {
            "id": d["id"],
            "title": d.get("title", ""),
            "raw_content": d.get("raw_content", ""),
            "entity_ids": d.get("entity_ids", []),
            "relationship_ids": d.get("relationship_ids", []),
            "content_embedding": d.get("content_embedding"),
        }
        for d in documents
    ]

    # Prepare entity rows
    ent_rows = [
        {
            "id": e["id"],
            "title": e.get("title", ""),
            "type": e.get("type", ""),
            "description": e.get("description", ""),
            "human_readable_id": e.get("human_readable_id", 0),
            "degree": e.get("degree", 0),
            "document_ids": e.get("document_ids", []),
            "community_ids": e.get("community_ids", []),
            "description_embedding": e.get("description_embedding"),
        }
        for e in entities
    ]

    # Prepare relationship rows
    rel_rows = [
        {
            "id": r["id"],
            "source_entity_id": r.get("source_entity_id", ""),
            "target_entity_id": r.get("target_entity_id", ""),
            "description": r.get("description", ""),
            "weight": r.get("weight", 1.0),
            "human_readable_id": r.get("human_readable_id", 0),
            "document_ids": r.get("document_ids", []),
        }
        for r in relationships
    ]

    # Prepare community rows
    comm_rows = [
        {
            "id": c["id"],
            "level": c.get("level", 0),
            "title": c.get("title"),
            "summary": c.get("summary"),
            "full_content": c.get("full_content"),
            "rating": c.get("rating"),
            "rating_explanation": c.get("rating_explanation"),
            "entity_ids": c.get("entity_ids", []),
            "relationship_ids": c.get("relationship_ids", []),
            "document_ids": c.get("document_ids", []),
            "parent_community_id": c.get("parent_community_id"),
            "child_community_ids": c.get("child_community_ids", []),
            "full_content_embedding": c.get("full_content_embedding"),
        }
        for c in communities
    ]

    logger.info("Writing to Spanner: %d docs, %d entities, %d rels, %d communities",
                len(doc_rows), len(ent_rows), len(rel_rows), len(comm_rows))

    spanner.bulk_write_documents(cfg, doc_rows)
    spanner.bulk_write_entities(cfg, ent_rows)
    spanner.bulk_write_relationships(cfg, rel_rows)
    spanner.bulk_write_communities(cfg, comm_rows)

    logger.info("Spanner sync complete")
