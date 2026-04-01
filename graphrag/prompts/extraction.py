"""Entity and relationship extraction prompt for telecom call transcripts."""

from __future__ import annotations

ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "PRODUCT",
    "SERVICE",
    "ISSUE",
    "LOCATION",
    "CONCEPT",
    "EVENT",
]

EXTRACTION_SYSTEM_INSTRUCTION = """\
You are an expert at extracting structured knowledge from telecom customer \
service call transcripts. Your task is to identify all meaningful entities \
and relationships mentioned in the conversation.

Entity types to extract: {entity_types}

Guidelines:
- Extract every distinct entity mentioned, including the customer, agent, \
products, services, technical issues, plans, promotions, locations, etc.
- Normalize entity names: use UPPERCASE for all entity names.
- For each entity, provide a concise but informative description based on \
what the transcript says about it.
- For each relationship, describe how the two entities are connected in the \
context of this transcript.
- Assign a weight between 0.0 and 1.0 to each relationship reflecting its \
importance in the transcript.
- Be thorough: it is better to extract too many entities than too few.\
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract all entities and relationships from the following telecom customer \
service call transcript.

---TRANSCRIPT START---
{text}
---TRANSCRIPT END---

Return the result as JSON with two arrays: "entities" and "relationships".\
"""

EXTRACTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "entities": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {
                        "type": "STRING",
                        "description": "The canonical name of the entity (UPPERCASE).",
                    },
                    "type": {
                        "type": "STRING",
                        "description": "One of: " + ", ".join(ENTITY_TYPES),
                    },
                    "description": {
                        "type": "STRING",
                        "description": "A concise description of the entity based on the transcript.",
                    },
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source": {
                        "type": "STRING",
                        "description": "The name of the source entity (UPPERCASE).",
                    },
                    "target": {
                        "type": "STRING",
                        "description": "The name of the target entity (UPPERCASE).",
                    },
                    "description": {
                        "type": "STRING",
                        "description": "A description of the relationship.",
                    },
                    "weight": {
                        "type": "NUMBER",
                        "description": "Importance weight from 0.0 to 1.0.",
                    },
                },
                "required": ["source", "target", "description", "weight"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


def format_extraction_prompt(text: str) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(text=text)


def format_system_instruction() -> str:
    return EXTRACTION_SYSTEM_INSTRUCTION.format(
        entity_types=", ".join(ENTITY_TYPES)
    )
