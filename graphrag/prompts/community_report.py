"""Community report generation prompt."""

from __future__ import annotations

COMMUNITY_REPORT_SYSTEM_INSTRUCTION = """\
You are an expert analyst producing structured community reports from \
knowledge graph data. Each community is a cluster of related entities \
and relationships extracted from telecom customer service call transcripts. \
Your task is to produce a report that summarizes the community's key themes, \
findings, and overall importance.\
"""

COMMUNITY_REPORT_TEMPLATE = """\
Write a comprehensive report for the following community of entities and \
relationships extracted from telecom customer service call transcripts.

{context}

The report should include:
- A short, descriptive title
- A summary paragraph
- An importance rating from 0 to 10 (where 10 is the most important)
- An explanation for the rating
- A list of key findings, each with a summary and detailed explanation

Focus on patterns, recurring issues, and actionable insights.\
"""

COMMUNITY_REPORT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {
            "type": "STRING",
            "description": "A short descriptive title for this community.",
        },
        "summary": {
            "type": "STRING",
            "description": "A summary paragraph of the community.",
        },
        "rating": {
            "type": "NUMBER",
            "description": "Importance rating from 0 to 10.",
        },
        "rating_explanation": {
            "type": "STRING",
            "description": "Explanation for the importance rating.",
        },
        "findings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "summary": {
                        "type": "STRING",
                        "description": "Short summary of the finding.",
                    },
                    "explanation": {
                        "type": "STRING",
                        "description": "Detailed explanation of the finding.",
                    },
                },
                "required": ["summary", "explanation"],
            },
        },
    },
    "required": ["title", "summary", "rating", "rating_explanation", "findings"],
}


def format_community_context(
    entities: list[dict],
    relationships: list[dict],
    sub_community_reports: list[dict] | None = None,
) -> str:
    """Build a context string from community members."""
    parts: list[str] = []

    if sub_community_reports:
        parts.append("## Sub-Community Reports")
        for report in sub_community_reports:
            parts.append(
                f"### {report.get('title', 'Untitled')}\n"
                f"{report.get('summary', '')}\n"
                f"Rating: {report.get('rating', 'N/A')}\n"
            )

    if entities:
        parts.append("## Entities")
        for e in entities:
            parts.append(
                f"- **{e['title']}** (type: {e['type']}, degree: {e.get('degree', 0)}): "
                f"{e.get('description', '')}"
            )

    if relationships:
        parts.append("## Relationships")
        for r in relationships:
            parts.append(
                f"- {r['source']} -> {r['target']} "
                f"(weight: {r.get('weight', 1.0)}): {r.get('description', '')}"
            )

    return "\n\n".join(parts)


def format_community_report_prompt(context: str) -> str:
    return COMMUNITY_REPORT_TEMPLATE.format(context=context)
