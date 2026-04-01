from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Node types ───────────────────────────────────────────────────────────


class NodeType(str, Enum):
    CALL = "Call"
    CUSTOMER = "Customer"
    AGENT = "Agent"
    PROBLEM = "Problem"
    PRODUCT = "Product"
    SERVICE = "Service"
    SOLUTION = "Solution"
    FEEDBACK = "Feedback"


class RelationshipType(str, Enum):
    INITIATED = "INITIATED"
    RELATES_TO = "RELATES_TO"
    AFFECTS = "AFFECTS"
    MENTIONS = "MENTIONS"
    PROVIDED = "PROVIDED"
    RESOLVES = "RESOLVES"
    ABOUT = "ABOUT"
    EXPRESSED_SENTIMENT_TOWARD = "EXPRESSED_SENTIMENT_TOWARD"
    HANDLED_BY = "HANDLED_BY"
    RESULTED_IN = "RESULTED_IN"
    REFERENCES = "REFERENCES"


# ── Extracted node (flat representation for BQ storage) ──────────────────


class RawNode(BaseModel):
    """A node extracted from a single transcript, before entity resolution."""

    id: str
    document_id: str
    node_type: str
    name: str
    # Call
    call_category: str | None = None
    call_outcome: str | None = None
    call_timestamp: str | None = None
    # Customer
    customer_id: str | None = None
    customer_type: str | None = None
    overall_sentiment: str | None = None
    # Agent
    agent_id: str | None = None
    agent_role: str | None = None
    # Problem
    issue_type: str | None = None
    severity: str | None = None
    issue_description: str | None = None
    # Product
    product_name: str | None = None
    product_type: str | None = None
    # Service
    service_name: str | None = None
    service_category: str | None = None
    # Solution
    solution_type: str | None = None
    resolution_status: str | None = None
    # Feedback
    feedback_type: str | None = None
    feedback_sentiment: str | None = None
    # Generic
    description: str | None = None


class RawRelationship(BaseModel):
    """A relationship extracted from a single transcript."""

    id: str
    document_id: str
    relationship_type: str
    source_name: str
    target_name: str
    description: str | None = None
    weight: float = 1.0


# ── Merged node (after entity resolution) ───────────────────────────────


class MergedNode(BaseModel):
    """Canonical node after Splink entity resolution."""

    id: str
    node_type: str
    name: str
    document_ids: list[str] = Field(default_factory=list)
    # Call
    call_category: str | None = None
    call_outcome: str | None = None
    call_timestamp: str | None = None
    # Customer
    customer_id: str | None = None
    customer_type: str | None = None
    overall_sentiment: str | None = None
    # Agent
    agent_id: str | None = None
    agent_role: str | None = None
    # Problem
    issue_type: str | None = None
    severity: str | None = None
    issue_description: str | None = None
    # Product
    product_name: str | None = None
    product_type: str | None = None
    # Service
    service_name: str | None = None
    service_category: str | None = None
    # Solution
    solution_type: str | None = None
    resolution_status: str | None = None
    # Feedback
    feedback_type: str | None = None
    feedback_sentiment: str | None = None
    # Generic
    description: str | None = None


class MergedRelationship(BaseModel):
    """Canonical relationship after entity resolution remapping."""

    id: str
    relationship_type: str
    source_node_id: str
    target_node_id: str
    description: str | None = None
    weight: float = 1.0
    document_ids: list[str] = Field(default_factory=list)
