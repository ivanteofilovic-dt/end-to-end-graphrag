from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str = ""
    raw_content: str
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    id: str
    title: str
    type: str
    description: str = ""
    human_readable_id: int = 0
    degree: int = 0
    document_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    id: str
    source: str
    target: str
    source_entity_id: str = ""
    target_entity_id: str = ""
    description: str = ""
    weight: float = 1.0
    human_readable_id: int = 0
    document_ids: list[str] = Field(default_factory=list)


class Community(BaseModel):
    id: str
    level: int
    title: str | None = None
    summary: str | None = None
    full_content: str | None = None
    rating: float | None = None
    rating_explanation: str | None = None
    findings: list[dict] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    parent_community_id: str | None = None
    child_community_ids: list[str] = Field(default_factory=list)
