"""Interactive text-to-GQL querying over the Spanner KnowledgeGraph.

Uses LangChain's SpannerGraphQAChain to translate natural-language questions
into GQL, execute them against Spanner Graph, and return LLM-synthesized answers.

The graph uses DYNAMIC LABEL, so the actual label values live in data columns
rather than in the property-graph DDL. We query distinct node_type /
relationship_type values at startup and inject them into the GQL generation
prompt so the LLM knows which labels exist.
"""

from __future__ import annotations

import logging
from typing import Any

from google.cloud import spanner
from langchain_core.prompts import PromptTemplate
from langchain_google_spanner import SpannerGraphStore, SpannerGraphQAChain
from langchain_google_vertexai import ChatVertexAI

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)

GQL_PROMPT_TEMPLATE = """\
Create a Spanner Graph GQL query for the question using the schema and the
available label values listed below.

This graph uses DYNAMIC LABEL — the label in a MATCH pattern must exactly match
one of the values from the "Available node labels" or "Available edge labels"
lists. Labels are **case-sensitive** and lowercase.

Available node labels (use these as node labels in MATCH):
{node_labels}

Available edge labels (use these as edge labels in MATCH):
{edge_labels}

Important GQL syntax rules:
- WHERE clauses go INSIDE the MATCH pattern parentheses:
  MATCH (n:label WHERE n.prop = 'value')
  Do NOT put WHERE after the MATCH clause.
- Path quantification uses curly braces after the arrow: -[e:label]->{{1, 3}}
  NOT the Cypher star syntax [e:label*1..3]
- Always start with GRAPH <graphname>
- Always alias RETURN values

Example queries:

GRAPH KnowledgeGraph
MATCH (c:customer WHERE c.overall_sentiment = 'negative')
RETURN c.name AS customer_name, c.customer_id AS customer_id

GRAPH KnowledgeGraph
MATCH (a:agent)-[r:handled]->(call:call)
RETURN a.name AS agent_name, call.call_category AS category

GRAPH KnowledgeGraph
MATCH (c:customer)-[r:reported]->(p:problem)
RETURN c.name AS customer, p.name AS problem, p.severity AS severity

GRAPH KnowledgeGraph
MATCH (p:problem)-[r:resolved_by]->(s:solution)
RETURN p.name AS problem, s.name AS solution, s.resolution_status AS status

GRAPH KnowledgeGraph
MATCH (c:call)
RETURN c.call_category AS category, COUNT(*) AS call_count
ORDER BY call_count DESC

Question: {question}
Schema: {schema}

Do not include any explanations or apologies.
Do not prefix the query with `gql`.
Do not include any backticks.
Start with GRAPH <graphname>.
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""


def _query_distinct_labels(
    graph: SpannerGraphStore,
) -> tuple[list[str], list[str]]:
    """Query the Nodes/Relationships tables for actual DYNAMIC LABEL values."""
    node_labels: list[str] = []
    edge_labels: list[str] = []
    try:
        rows: list[dict[str, Any]] = graph.query(
            "SELECT DISTINCT node_type FROM Nodes ORDER BY node_type"
        )
        node_labels = [r["node_type"] for r in rows]
    except Exception:
        logger.warning("Could not query distinct node labels")

    try:
        rows = graph.query(
            "SELECT DISTINCT relationship_type FROM Relationships "
            "ORDER BY relationship_type"
        )
        edge_labels = [r["relationship_type"] for r in rows]
    except Exception:
        logger.warning("Could not query distinct edge labels")

    return node_labels, edge_labels


def _build_gql_prompt(
    node_labels: list[str], edge_labels: list[str]
) -> PromptTemplate:
    """Build the GQL generation prompt with actual label values baked in."""
    return PromptTemplate(
        template=GQL_PROMPT_TEMPLATE,
        input_variables=["question", "schema"],
        partial_variables={
            "node_labels": "\n".join(f"  - {l}" for l in node_labels)
            or "  (none found)",
            "edge_labels": "\n".join(f"  - {l}" for l in edge_labels)
            or "  (none found)",
        },
    )


def build_chain(cfg: GraphRAGConfig) -> SpannerGraphQAChain:
    """Construct a SpannerGraphQAChain from the pipeline config."""
    client = spanner.Client(project=cfg.gcp.project_id)
    graph = SpannerGraphStore(
        instance_id=cfg.spanner.instance_id,
        database_id=cfg.spanner.database_id,
        graph_name=cfg.spanner.graph_name,
        client=client,
    )

    node_labels, edge_labels = _query_distinct_labels(graph)
    logger.info(
        "Connected to Spanner Graph %s — %d node labels, %d edge labels",
        cfg.spanner.graph_name,
        len(node_labels),
        len(edge_labels),
    )
    logger.info("Node labels: %s", node_labels)
    logger.info("Edge labels: %s", edge_labels)

    gql_prompt = _build_gql_prompt(node_labels, edge_labels)

    llm = ChatVertexAI(
        model_name=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_output_tokens=cfg.llm.max_output_tokens,
        project=cfg.gcp.project_id,
        location=cfg.gcp.location,
    )

    chain = SpannerGraphQAChain.from_llm(
        llm,
        graph=graph,
        gql_prompt=gql_prompt,
        allow_dangerous_requests=True,
        verify_gql=False,
        return_intermediate_steps=True,
        verbose=True,
    )
    return chain


def _print_result(result: dict[str, Any]) -> None:
    """Print the chain result including the generated GQL."""
    for step in result.get("intermediate_steps", []):
        if "generated_query" in step:
            print(f"\nGenerated GQL:\n  {step['generated_query']}")
        if "context" in step:
            print(f"\nGraph results ({len(step['context'])} rows):")
            for row in step["context"][:5]:
                print(f"  {row}")
            if len(step["context"]) > 5:
                print(f"  ... and {len(step['context']) - 5} more")
    print(f"\nAnswer: {result['result']}\n")


def ask(cfg: GraphRAGConfig, question: str) -> str:
    """Run a single question against the knowledge graph and return the answer."""
    chain = build_chain(cfg)
    result = chain.invoke(question)
    _print_result(result)
    return result["result"]


def interactive(cfg: GraphRAGConfig) -> None:
    """Start an interactive REPL for querying the knowledge graph."""
    chain = build_chain(cfg)
    print("Knowledge Graph QA (type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in ("exit", "quit"):
            break

        try:
            result = chain.invoke(question)
            _print_result(result)
        except Exception:
            logger.exception("Query failed")
            print("Error: query failed, see logs above.\n")
