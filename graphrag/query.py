"""Interactive text-to-GQL querying over the Spanner KnowledgeGraph.

Uses LangChain's SpannerGraphQAChain to translate natural-language questions
into GQL, execute them against Spanner Graph, and return LLM-synthesized answers.
"""

from __future__ import annotations

import logging

from langchain_google_spanner import SpannerGraphStore, SpannerGraphQAChain
from langchain_google_vertexai import ChatVertexAI

from graphrag.config import GraphRAGConfig

logger = logging.getLogger(__name__)


def build_chain(cfg: GraphRAGConfig) -> SpannerGraphQAChain:
    """Construct a SpannerGraphQAChain from the pipeline config."""
    graph = SpannerGraphStore(
        instance_id=cfg.spanner.instance_id,
        database_id=cfg.spanner.database_id,
        graph_name=cfg.spanner.graph_name,
        project_id=cfg.gcp.project_id,
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
        allow_dangerous_requests=True,
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
