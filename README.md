# Telecom Knowledge Graph Indexing Pipeline

Builds a knowledge graph from telecom customer call transcripts stored in BigQuery.
Extracts typed entities and relationships using Gemini-3 batch API, deduplicates
them with Splink probabilistic entity resolution, and writes the merged graph to
Spanner Graph for querying via GQL / text-to-cypher.

## Architecture

```
BigQuery (source)
  │
  ▼
Step 1 ─ Load Documents ──────────► BQ: documents
  │
  ▼
Step 2 ─ Extract Graph ───────────► BQ: raw_nodes, raw_relationships
          (Gemini-3 Batch API)
  │
  ▼
Step 3 ─ Entity Resolution ───────► BQ: merged_nodes, merged_relationships
          (Splink 4 + DuckDB)
  │
  ▼
Step 4 ─ Write to Spanner ────────► Spanner Graph: KnowledgeGraph
                                        │
                                        ▼
                                   Text-to-GQL queries
                                   (SpannerGraphQAChain)
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A GCP project with the following APIs enabled:
  - BigQuery
  - Vertex AI (for Gemini batch predictions)
  - Spanner
- Application Default Credentials configured (`gcloud auth application-default login`)

### GCP resources you need to create beforehand

| Resource | Description |
|----------|-------------|
| BigQuery source table | Must contain columns `data_id` (STRING), `full_conversation` (STRING), `conversation_date` (STRING/DATE) |
| BigQuery intermediate dataset | Created automatically by the pipeline (default: `graphrag`) |
| Spanner instance + database | The pipeline creates tables and the property graph schema, but the instance and empty database must exist |

## Installation

```bash
uv sync
```

## Configuration

Copy and edit `config.yaml`:

```yaml
gcp:
  project_id: "your-gcp-project"
  location: "us-central1"

bigquery:
  source_dataset: "calls"           # dataset containing your transcripts
  source_table: "transcripts"       # table name
  intermediate_dataset: "graphrag"  # pipeline working dataset (auto-created)

spanner:
  instance_id: "graphrag-instance"
  database_id: "graphrag-db"

llm:
  model: "gemini-3-flash"
  temperature: 0.0
  max_output_tokens: 8192

splink:
  match_weight_threshold: 6.0   # minimum match weight for Splink predictions
  cluster_threshold: 0.95       # probability threshold for clustering

pipeline:
  batch_size: 500
```

## Usage

### Run the full pipeline

```bash
uv run graphrag --config config.yaml
```

### Run a single step

```bash
uv run graphrag --step 1          # Load documents only
uv run graphrag --step 2          # Extract graph only
uv run graphrag --step 3          # Entity resolution only
uv run graphrag --step 4          # Write to Spanner only
```

### Resume from a specific step

```bash
uv run graphrag --from-step 3     # Run steps 3 and 4
```

### Test run with limited data

```bash
uv run graphrag --step 2 --max-rows 100
```

This limits step 2 (extraction) to the first 100 documents -- useful for
validating the prompt and pipeline before processing the full dataset.

## Pipeline steps

### Step 1: Load Documents

Reads transcripts from the BigQuery source table and writes them to an
intermediate `documents` table. Deduplicates by `data_id`.

### Step 2: Extract Graph (Gemini-3 Batch API)

For each document, builds a `GenerateContentRequest` with a telecom-specific
extraction prompt and submits it as a Gemini batch job (BigQuery-to-BigQuery).
The model returns structured JSON with typed nodes and relationships.

**Extracted node types:**

| Node Type | Key Attributes |
|-----------|---------------|
| Call | call_category, call_outcome, timestamp |
| Customer | customer_id, customer_type, overall_sentiment |
| Agent | agent_id, role |
| Problem | issue_type, severity, description |
| Product | product_name, product_type |
| Service | service_name, service_category |
| Solution | solution_type, resolution_status |
| Feedback | feedback_type, sentiment |

**Extracted relationship types:**

INITIATED, RELATES_TO, AFFECTS, MENTIONS, PROVIDED, RESOLVES, ABOUT,
EXPRESSED_SENTIMENT_TOWARD, HANDLED_BY, RESULTED_IN, REFERENCES

### Step 3: Entity Resolution (Splink)

Runs probabilistic deduplication per entity type using Splink 4 with a local
DuckDB backend. Call nodes are passed through without resolution (1:1 with
transcripts). For all other types, the pipeline:

1. Trains a Splink model unsupervised (random sampling + EM)
2. Predicts pairwise matches above the configured weight threshold
3. Clusters matched records into canonical entities
4. Remaps all relationships to point to canonical node IDs
5. Deduplicates relationships sharing the same (source, target, type)

Falls back to simple name-based merging if Splink training fails for a type.

### Step 4: Write to Spanner Graph

Creates the Spanner schema (Nodes table, Relationships table, secondary indexes,
and a `KnowledgeGraph` property graph) then bulk-writes all merged data.

## Spanner Graph schema

```sql
-- All node types in a single table, discriminated by node_type
CREATE TABLE Nodes (
    id STRING(64) NOT NULL,
    node_type STRING(64) NOT NULL,
    name STRING(MAX),
    -- type-specific attributes as nullable columns
    ...
) PRIMARY KEY (id)

CREATE TABLE Relationships (
    id STRING(64) NOT NULL,
    relationship_type STRING(64) NOT NULL,
    source_node_id STRING(64) NOT NULL,
    target_node_id STRING(64) NOT NULL,
    description STRING(MAX),
    weight FLOAT64,
    document_ids ARRAY<STRING(64)>
) PRIMARY KEY (id)

CREATE OR REPLACE PROPERTY GRAPH KnowledgeGraph
    NODE TABLES (Nodes KEY (id))
    EDGE TABLES (
        Relationships KEY (id)
            SOURCE KEY (source_node_id) REFERENCES Nodes(id)
            DESTINATION KEY (target_node_id) REFERENCES Nodes(id)
    )
```

## Querying the graph

Once the pipeline completes, query the knowledge graph with GQL:

```sql
-- Count calls by category
GRAPH KnowledgeGraph
MATCH (call {node_type: 'Call'})
RETURN call.call_category, COUNT(*) AS total
ORDER BY total DESC

-- Find unresolved problems and affected products
GRAPH KnowledgeGraph
MATCH (p {node_type: 'Problem'})-[r {relationship_type: 'AFFECTS'}]->(prod {node_type: 'Product'})
WHERE p.severity = 'high'
RETURN p.issue_type, prod.product_name, COUNT(*) AS occurrences
ORDER BY occurrences DESC

-- Customer sentiment toward services
GRAPH KnowledgeGraph
MATCH (c {node_type: 'Customer'})-[r {relationship_type: 'EXPRESSED_SENTIMENT_TOWARD'}]->(s {node_type: 'Service'})
RETURN s.service_name, c.overall_sentiment, COUNT(*) AS cnt
ORDER BY cnt DESC
```

For natural-language queries, use `SpannerGraphQAChain` from the
`langchain-google-spanner` package to translate questions into GQL automatically.

## Project structure

```
├── config.yaml                          # Pipeline configuration
├── main.py                              # CLI entrypoint
├── pyproject.toml
├── graphrag/
│   ├── config.py                        # Pydantic config models
│   ├── models.py                        # Node/relationship data models
│   ├── batch/
│   │   ├── client.py                    # Gemini Batch API client
│   │   └── request_builder.py           # GenerateContentRequest builder
│   ├── pipeline/
│   │   ├── load_documents.py            # Step 1
│   │   ├── extract_graph.py             # Step 2
│   │   ├── entity_resolution.py         # Step 3
│   │   └── write_spanner.py             # Step 4
│   ├── prompts/
│   │   └── extraction.py                # Extraction prompt + JSON schema
│   └── storage/
│       ├── bigquery.py                  # BigQuery helpers
│       └── spanner.py                   # Spanner schema + bulk writes
```
