"""Entity and relationship extraction prompt for telecom call transcripts.

Defines the system instruction, user prompt template, and Gemini structured
output schema for extracting 8 node types and 11 relationship types.
"""

from __future__ import annotations

NODE_TYPES = [
    "Call",
    "Customer",
    "Agent",
    "Problem",
    "Product",
    "Service",
    "Solution",
    "Feedback",
]

RELATIONSHIP_TYPES = [
    "INITIATED",
    "RELATES_TO",
    "AFFECTS",
    "MENTIONS",
    "PROVIDED",
    "RESOLVES",
    "ABOUT",
    "EXPRESSED_SENTIMENT_TOWARD",
    "HANDLED_BY",
    "RESULTED_IN",
    "REFERENCES",
]

EXTRACTION_SYSTEM_INSTRUCTION = """\
You are an expert knowledge-graph builder for telecom customer service data. \
Given a customer call transcript you must extract all entities (nodes) and \
relationships (edges) that are explicitly or strongly implied in the \
conversation.

### Node types to extract

1. **Call** -- the interaction itself.
   Attributes: call_category (billing, technical support, cancellation, \
general inquiry, etc.), call_outcome (resolved, unresolved, follow-up \
needed, escalated), timestamp (ISO-8601 if mentioned).

2. **Customer** -- the person calling.
   Attributes: customer_id (if mentioned), customer_type (new, existing, \
business, private), overall_sentiment (positive, neutral, negative, mixed).

3. **Agent** -- the company representative.
   Attributes: agent_id (if mentioned), role (if mentioned).

4. **Problem** -- each distinct issue raised by the customer.
   Attributes: issue_type (concise label), severity (low, medium, high, \
critical), description (brief normalized description).

5. **Product** -- any product explicitly mentioned.
   Attributes: product_name, product_type.

6. **Service** -- any service or subscription mentioned.
   Attributes: service_name, service_category.

7. **Solution** -- what the company did or proposed.
   Attributes: solution_type (concise label), resolution_status (applied, \
pending, rejected).

8. **Feedback** -- explicit feedback from the customer.
   Attributes: feedback_type (complaint, praise, suggestion), \
sentiment (positive, neutral, negative).

### Relationship types to extract

Use EXACTLY these relationship labels:
- Customer **INITIATED** Call
- Call **RELATES_TO** Problem
- Problem **AFFECTS** Product
- Problem **AFFECTS** Service
- Call **MENTIONS** Product
- Call **MENTIONS** Service
- Agent **PROVIDED** Solution
- Solution **RESOLVES** Problem
- Customer **PROVIDED** Feedback
- Feedback **ABOUT** Product or Service
- Customer **EXPRESSED_SENTIMENT_TOWARD** Product or Service
- Agent **HANDLED_BY** Call  (the agent handled the call)
- Solution **RESULTED_IN** Call  (solution resulted from the call)
- Call **REFERENCES** Problem  (alternative to RELATES_TO if indirect)

### Guidelines

- Extract EVERY entity mentioned, even if only implied.
- Use a short, normalised uppercase NAME for each node (e.g. "INTERNET \
SERVICE", "BILLING ISSUE").
- Each transcript should produce exactly ONE Call node and at most ONE \
Customer node.
- Assign a weight between 0.0 and 1.0 to each relationship reflecting its \
importance.
- Return valid JSON matching the provided schema.\
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract all nodes and relationships from the following telecom customer \
call transcript.

---TRANSCRIPT START---
{text}
---TRANSCRIPT END---

Return the result as JSON with two arrays: "nodes" and "relationships".\
"""

# ── Gemini structured output schema ─────────────────────────────────────

_NODE_ATTRIBUTES_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "call_category": {"type": "STRING", "nullable": True},
        "call_outcome": {"type": "STRING", "nullable": True},
        "timestamp": {"type": "STRING", "nullable": True},
        "customer_id": {"type": "STRING", "nullable": True},
        "customer_type": {"type": "STRING", "nullable": True},
        "overall_sentiment": {"type": "STRING", "nullable": True},
        "agent_id": {"type": "STRING", "nullable": True},
        "role": {"type": "STRING", "nullable": True},
        "issue_type": {"type": "STRING", "nullable": True},
        "severity": {"type": "STRING", "nullable": True},
        "description": {"type": "STRING", "nullable": True},
        "product_name": {"type": "STRING", "nullable": True},
        "product_type": {"type": "STRING", "nullable": True},
        "service_name": {"type": "STRING", "nullable": True},
        "service_category": {"type": "STRING", "nullable": True},
        "solution_type": {"type": "STRING", "nullable": True},
        "resolution_status": {"type": "STRING", "nullable": True},
        "feedback_type": {"type": "STRING", "nullable": True},
        "sentiment": {"type": "STRING", "nullable": True},
    },
}

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "nodes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "node_type": {
                        "type": "STRING",
                        "description": "One of: " + ", ".join(NODE_TYPES),
                    },
                    "name": {
                        "type": "STRING",
                        "description": "Short normalised UPPERCASE name.",
                    },
                    "attributes": _NODE_ATTRIBUTES_SCHEMA,
                },
                "required": ["node_type", "name", "attributes"],
            },
        },
        "relationships": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "relationship_type": {
                        "type": "STRING",
                        "description": "One of: " + ", ".join(RELATIONSHIP_TYPES),
                    },
                    "source": {
                        "type": "STRING",
                        "description": "Name of the source node (UPPERCASE).",
                    },
                    "target": {
                        "type": "STRING",
                        "description": "Name of the target node (UPPERCASE).",
                    },
                    "description": {
                        "type": "STRING",
                        "description": "Brief description of the relationship.",
                    },
                    "weight": {
                        "type": "NUMBER",
                        "description": "Importance weight from 0.0 to 1.0.",
                    },
                },
                "required": [
                    "relationship_type",
                    "source",
                    "target",
                    "description",
                    "weight",
                ],
            },
        },
    },
    "required": ["nodes", "relationships"],
}


def format_extraction_prompt(text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(text=text)


def format_system_instruction() -> str:
    return EXTRACTION_SYSTEM_INSTRUCTION
