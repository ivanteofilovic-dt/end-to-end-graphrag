"""Description summarization prompt for merging multiple entity/relationship descriptions."""

from __future__ import annotations

SUMMARIZATION_SYSTEM_INSTRUCTION = """\
You are an expert at synthesizing information. Given multiple descriptions \
of the same entity or relationship extracted from different sources, merge \
them into a single comprehensive description that captures all important \
details without redundancy.\
"""

ENTITY_SUMMARIZATION_TEMPLATE = """\
The following descriptions all refer to the same entity named "{name}" \
(type: {entity_type}). Merge them into a single, comprehensive description.

Descriptions:
{descriptions}

Provide a single merged description.\
"""

RELATIONSHIP_SUMMARIZATION_TEMPLATE = """\
The following descriptions all refer to the same relationship between \
"{source}" and "{target}". Merge them into a single, comprehensive \
description.

Descriptions:
{descriptions}

Provide a single merged description.\
"""

SUMMARIZATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {
            "type": "STRING",
            "description": "The merged, comprehensive description.",
        },
    },
    "required": ["description"],
}


def format_entity_summarization_prompt(
    name: str,
    entity_type: str,
    descriptions: list[str],
) -> str:
    numbered = "\n".join(
        f"{i + 1}. {desc}" for i, desc in enumerate(descriptions)
    )
    return ENTITY_SUMMARIZATION_TEMPLATE.format(
        name=name,
        entity_type=entity_type,
        descriptions=numbered,
    )


def format_relationship_summarization_prompt(
    source: str,
    target: str,
    descriptions: list[str],
) -> str:
    numbered = "\n".join(
        f"{i + 1}. {desc}" for i, desc in enumerate(descriptions)
    )
    return RELATIONSHIP_SUMMARIZATION_TEMPLATE.format(
        source=source,
        target=target,
        descriptions=numbered,
    )
