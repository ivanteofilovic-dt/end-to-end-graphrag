"""Community report generation prompt for telecom knowledge graph communities.

Defines the system instruction, user prompt template, Gemini structured output
schema, and formatting helpers for turning community entities/relationships into
a readable prompt that the LLM can summarize.
"""

from __future__ import annotations

from typing import Any

COMMUNITY_REPORT_SYSTEM_INSTRUCTION = """\
You are an expert analyst creating a community report from a telecom customer \
service knowledge graph.

You are given a set of entities and relationships that form a community \
(cluster) in the graph.  Your task is to create a comprehensive analytical \
report about this community.

The report MUST include:

1. **title** — A short, specific title (≤12 words) that captures the \
community's primary theme.
2. **summary** — 2-4 sentences describing the community's key themes, \
patterns, and significance for telecom operations.
3. **findings** — 3-8 key insights, each with a concise ``summary`` \
(one sentence) and a longer ``explanation`` (2-4 sentences) that references \
specific entity names and relationship details from the input.
4. **rating** — A float from 1.0 to 10.0 reflecting how significant this \
community is for understanding telecom customer service patterns.
5. **rating_explanation** — 1-2 sentences justifying the score.

### Guidelines

- Focus on patterns, recurring issues, and actionable insights.
- Highlight connections between problems, products/services, and resolutions.
- Note customer sentiment trends and agent effectiveness where visible.
- Be specific — reference entity names and relationship details.
- Return valid JSON matching the provided schema.\
"""

COMMUNITY_REPORT_PROMPT_TEMPLATE = """\
Analyze the following community of entities and relationships extracted from \
a telecom customer service knowledge graph and generate a comprehensive \
community report.

---ENTITIES---
{entities}

---RELATIONSHIPS---
{relationships}

Generate a report as JSON matching the required schema.\
"""

# ── Gemini structured output schema ─────────────────────────────────────

COMMUNITY_REPORT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {
            "type": "STRING",
            "description": "Short title capturing the community's primary theme.",
        },
        "summary": {
            "type": "STRING",
            "description": "2-4 sentence summary of key themes and significance.",
        },
        "findings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "summary": {
                        "type": "STRING",
                        "description": "One-sentence insight summary.",
                    },
                    "explanation": {
                        "type": "STRING",
                        "description": "2-4 sentence explanation with entity references.",
                    },
                },
                "required": ["summary", "explanation"],
            },
        },
        "rating": {
            "type": "NUMBER",
            "description": "Significance rating from 1.0 to 10.0.",
        },
        "rating_explanation": {
            "type": "STRING",
            "description": "1-2 sentence justification for the rating.",
        },
    },
    "required": ["title", "summary", "findings", "rating", "rating_explanation"],
}


# ── Formatting helpers ───────────────────────────────────────────────────

_SKIP_KEYS = frozenset({"id", "name", "node_type", "document_ids"})


def format_entity(node: dict[str, Any]) -> str:
    """Format a single node into a human-readable line for the prompt."""
    name = node.get("name", "UNKNOWN")
    node_type = node.get("node_type", "Unknown")

    attrs: list[str] = []
    for key, value in node.items():
        if key in _SKIP_KEYS or not value:
            continue
        attrs.append(f"{key}: {value}")

    attr_str = f" ({', '.join(attrs)})" if attrs else ""
    return f"- {name} [{node_type}]{attr_str}"


def format_relationship(
    rel: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
) -> str:
    """Format a single relationship into a human-readable line."""
    src = node_lookup.get(rel.get("source_node_id", ""), {})
    tgt = node_lookup.get(rel.get("target_node_id", ""), {})
    src_name = src.get("name", "UNKNOWN")
    tgt_name = tgt.get("name", "UNKNOWN")
    rel_type = rel.get("relationship_type", "RELATED_TO")
    desc = rel.get("description", "")
    weight = rel.get("weight", 1.0)

    line = f"- {src_name} --[{rel_type}]--> {tgt_name}"
    if desc:
        line += f": {desc}"
    line += f" (weight: {weight:.1f})"
    return line


def format_community_prompt(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    node_lookup: dict[str, dict[str, Any]],
) -> str:
    """Build the user prompt for a single community."""
    entity_lines = "\n".join(format_entity(e) for e in entities) or "(none)"
    rel_lines = (
        "\n".join(format_relationship(r, node_lookup) for r in relationships)
        or "(none)"
    )
    return COMMUNITY_REPORT_PROMPT_TEMPLATE.format(
        entities=entity_lines,
        relationships=rel_lines,
    )


def format_system_instruction() -> str:
    return COMMUNITY_REPORT_SYSTEM_INSTRUCTION
