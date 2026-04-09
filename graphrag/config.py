from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class GCPConfig(BaseModel):
    project_id: str
    location: str = "us-central1"


class BigQueryConfig(BaseModel):
    source_dataset: str
    source_table: str
    intermediate_dataset: str = "graphrag"


class SpannerConfig(BaseModel):
    instance_id: str
    database_id: str
    graph_name: str = "KnowledgeGraph"


class LLMConfig(BaseModel):
    model: str = "gemini-3-flash"
    temperature: float = 0.0
    max_output_tokens: int = 8192


class SplinkConfig(BaseModel):
    match_weight_threshold: float = 2.0
    cluster_threshold: float = 0.7
    max_pairs_per_type: int = 10_000_000


class CommunityConfig(BaseModel):
    max_cluster_size: int = 10
    seed: int | None = 42


class PipelineConfig(BaseModel):
    batch_size: int = 500


class GraphRAGConfig(BaseModel):
    gcp: GCPConfig
    bigquery: BigQueryConfig
    spanner: SpannerConfig
    llm: LLMConfig = LLMConfig()
    splink: SplinkConfig = SplinkConfig()
    community: CommunityConfig = CommunityConfig()
    pipeline: PipelineConfig = PipelineConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraphRAGConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @property
    def intermediate_table(self) -> str:
        return f"{self.gcp.project_id}.{self.bigquery.intermediate_dataset}"

    def table_fqn(self, table_name: str) -> str:
        """Fully-qualified BigQuery table name in the intermediate dataset."""
        return f"{self.gcp.project_id}.{self.bigquery.intermediate_dataset}.{table_name}"

    def source_table_fqn(self) -> str:
        return (
            f"{self.gcp.project_id}"
            f".{self.bigquery.source_dataset}"
            f".{self.bigquery.source_table}"
        )
