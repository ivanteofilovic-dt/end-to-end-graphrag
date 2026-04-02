"""Interactive text-to-GQL querying over the Spanner KnowledgeGraph.

Uses LangChain's SpannerGraphQAChain to translate natural-language questions
into GQL, execute them against Spanner Graph, and return LLM-synthesized answers.
"""

from __future__ import annotations

import logging

from google.cloud import spanner
from langchain_core.prompts import PromptTemplate
from langchain_google_spanner import SpannerGraphStore, SpannerGraphQAChain
from langchain_google_vertexai import ChatVertexAI

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)

GQL_PROMPT = PromptTemplate(
    template="""\
Create a Spanner Graph GQL query for the question using the schema.

Important GQL syntax rules:
- WHERE clauses go INSIDE the MATCH pattern parentheses, e.g.
  MATCH (n:Label WHERE n.prop = 'value')
  NOT after the MATCH clause.
- Path quantification uses curly braces after the arrow: -[e:Edge]->{{1, 3}}
  NOT the Cypher star syntax [e:Edge*1..3]
- Property filters inside patterns use double curly braces: (n:Label {{id: 7}})
- Always start with GRAPH <graphname>
- Always alias RETURN values

Example queries against a KnowledgeGraph with DYNAMIC LABEL nodes/edges:

Find customers with negative sentiment:
GRAPH KnowledgeGraph
MATCH (c:customer WHERE c.overall_sentiment = 'negative')
RETURN c.name AS customer_name, c.customer_id AS customer_id

Find which agent handled a call:
GRAPH KnowledgeGraph
MATCH (a:agent)-[r:handled]->(call:call)
RETURN a.name AS agent_name, call.call_category AS category

Find problems reported by a specific customer:
GRAPH KnowledgeGraph
MATCH (c:customer WHERE c.name = 'John Smith')-[r:reported]->(p:problem)
RETURN p.name AS problem, p.severity AS severity, p.issue_type AS issue_type

Find solutions applied to problems:
GRAPH KnowledgeGraph
MATCH (p:problem)-[r:resolved_by]->(s:solution)
RETURN p.name AS problem, s.name AS solution, s.resolution_status AS status

Count calls by category:
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
""",
    input_variables=["question", "schema"],
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
    logger.info(
        "Connected to Spanner Graph %s (instance=%s, db=%s)",
        cfg.spanner.graph_name,
        cfg.spanner.instance_id,
        cfg.spanner.database_id,
    )

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
        gql_prompt=GQL_PROMPT,
        allow_dangerous_requests=True,
        verify_gql=False,
        verbose=True,
    )
    return chain


def ask(cfg: GraphRAGConfig, question: str) -> str:
    """Run a single question against the knowledge graph and return the answer."""
    chain = build_chain(cfg)
    result = chain.invoke(question)
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
            print(f"\nAnswer: {result['result']}\n")
        except Exception:
            logger.exception("Query failed")
            print("Error: query failed, see logs above.\n")
