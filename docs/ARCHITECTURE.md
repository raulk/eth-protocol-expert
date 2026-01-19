# Architecture

This document describes the system architecture of the Ethereum Protocol Intelligence System.

## High-level overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│                     CLI (query_cli.py) | REST API                        │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Query Processing                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Concept   │  │    Query     │  │    Cost     │  │   Agentic    │  │
│  │  Resolution │→ │  Expansion   │→ │   Router    │→ │  Retrieval   │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Retrieval                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │  Embedding  │  │   Vector     │  │   Hybrid    │  │  Reranking   │  │
│  │   Query     │→ │   Search     │→ │   Search    │→ │   (Cohere)   │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Generation                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Context   │  │    Prompt    │  │    LLM      │  │   Response   │  │
│  │  Assembly   │→ │  Building    │→ │ Generation  │→ │   Parsing    │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Validation                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Claim     │  │    Atomic    │  │     NLI     │  │   Support    │  │
│  │ Extraction  │→ │Decomposition │→ │ Inference   │→ │Classification│  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data flow

### Ingestion pipeline

```
GitHub (ethereum/EIPs)
        │
        ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   EIP Loader  │ ──▶ │  EIP Parser   │ ──▶ │    Chunker    │
│  (Git clone)  │     │ (Frontmatter) │     │  (Sections)   │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                                    ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   pgvector    │ ◀── │   Embedder    │ ◀── │    Chunks     │
│  (Storage)    │     │  (Voyage AI)  │     │   (Text)      │
└───────────────┘     └───────────────┘     └───────────────┘
```

### Query pipeline

```
User Query: "What is EIP-1559?"
        │
        ▼
┌───────────────┐
│    Concept    │  "1559" → "EIP-1559"
│   Resolution  │  "base fee" → "base_fee"
└───────────────┘
        │
        ▼
┌───────────────┐
│    Query      │  Add synonyms and related terms
│   Expansion   │
└───────────────┘
        │
        ▼
┌───────────────┐
│   Embedding   │  Query → [0.12, -0.34, ...] (1024 dims)
└───────────────┘
        │
        ▼
┌───────────────┐
│    Vector     │  SELECT * FROM chunks
│    Search     │  ORDER BY embedding <=> query_embedding
└───────────────┘  LIMIT 5
        │
        ▼
┌───────────────┐
│   Context     │  Format chunks with citations
│   Assembly    │  [1] EIP-1559, Abstract: ...
└───────────────┘
        │
        ▼
┌───────────────┐
│     LLM       │  Claude Sonnet 4
│  Generation   │  System prompt + Context + Query
└───────────────┘
        │
        ▼
┌───────────────┐
│  Validation   │  (Optional) NLI claim verification
│    (NLI)      │
└───────────────┘
        │
        ▼
Response with citations and validation flags
```

## Module architecture

### Core modules

```
src/
├── ingestion/                 # Phase 0, 6, 10
│   ├── eip_loader.py             # Clone and load EIPs from GitHub
│   ├── eip_parser.py             # Parse markdown, extract frontmatter
│   ├── acd_transcript_loader.py  # AllCoreDevs call transcripts
│   ├── arxiv_fetcher.py          # Academic papers from arXiv
│   ├── git_code_loader.py        # Ethereum client source code
│   ├── pdf_extractor.py          # PDF document extraction
│   └── quality_scorer.py         # Content quality assessment
│
├── chunking/                  # Phase 0, 1
│   ├── fixed_chunker.py          # Simple token-based splitting
│   ├── section_chunker.py        # Section-aware (respects headers)
│   └── code_chunker.py           # Function-level code splitting
│
├── embeddings/                # Phase 0
│   ├── voyage_embedder.py        # Voyage AI (voyage-3)
│   ├── local_embedder.py         # Local BGE model
│   └── code_embedder.py          # Code-specific (voyage-code-2)
│
├── storage/                   # Phase 0
│   └── pg_vector_store.py        # PostgreSQL + pgvector
│
├── retrieval/                 # Phase 0, 3
│   └── simple_retriever.py       # Vector similarity search
│
├── generation/                # Phase 0, 1, 2
│   ├── simple_generator.py       # Basic RAG generation
│   ├── cited_generator.py        # With inline citations
│   ├── validated_generator.py    # With NLI validation
│   └── synthesis_generator.py    # Multi-source synthesis
│
├── validation/                # Phase 2
│   ├── nli_validator.py          # NLI-based claim verification
│   └── claim_decomposer.py       # Atomic fact extraction
│
└── evidence/                  # Phase 1
    ├── evidence_span.py          # Immutable source references
    └── evidence_ledger.py        # Claim-to-evidence mapping
```

### Advanced modules

```
src/
├── concepts/                  # Phase 7
│   ├── alias_table.py            # Ethereum terminology mappings
│   ├── query_expander.py         # Add synonyms to queries
│   └── concept_resolver.py       # Canonical term resolution
│
├── agents/                    # Phase 8
│   ├── react_agent.py            # ReAct-style reasoning agent
│   ├── budget_enforcer.py        # Resource limits (retrievals, tokens)
│   ├── backtrack.py              # Dead-end detection
│   └── reflection.py             # Quality assessment
│
├── structured/                # Phase 9
│   ├── timeline_builder.py       # Temporal event ordering
│   ├── argument_mapper.py        # Pro/con analysis
│   ├── comparison_table.py       # Feature comparisons
│   └── dependency_view.py        # Dependency visualization
│
├── graph/                     # Phase 4, 11, 13
│   ├── eip_graph_builder.py      # EIP dependency graph
│   ├── cross_reference.py        # Cross-document links
│   ├── citation_graph.py         # Academic paper citations
│   ├── relationship_inferrer.py  # Semantic relationship inference
│   ├── confidence_scorer.py      # Relationship confidence scoring
│   ├── selective_traverser.py    # High-confidence path following
│   └── spec_impl_linker.py       # EIP-to-code mapping
│
├── ensemble/                  # Phase 12
│   ├── cost_router.py            # Model selection by cost/quality
│   ├── multi_model_runner.py     # Parallel model execution
│   └── conditional_trigger.py    # Ensemble activation logic
│
└── parsing/                   # Phase 13
    ├── treesitter_parser.py      # Go/Rust syntax parsing
    └── code_unit_extractor.py    # Function/struct extraction
```

## Database schema

### Documents table

Stores EIP metadata:

```sql
CREATE TABLE documents (
    document_id    TEXT PRIMARY KEY,      -- "eip-1559"
    eip_number     INTEGER,               -- 1559
    title          TEXT,                  -- "Fee Market Change"
    status         TEXT,                  -- "Final"
    type           TEXT,                  -- "Standards Track"
    category       TEXT,                  -- "Core"
    author         TEXT,                  -- "Vitalik Buterin"
    created_date   DATE,                  -- 2019-04-13
    requires       INTEGER[],             -- [1, 155]
    raw_content    TEXT,                  -- Full markdown
    git_commit     TEXT,                  -- Source commit hash
    indexed_at     TIMESTAMP              -- When indexed
);
```

### Chunks table

Stores text chunks with embeddings:

```sql
CREATE TABLE chunks (
    chunk_id       TEXT PRIMARY KEY,      -- UUID
    document_id    TEXT REFERENCES documents,
    content        TEXT,                  -- Chunk text
    section_path   TEXT,                  -- "Specification > Base Fee"
    chunk_index    INTEGER,               -- Position in document
    token_count    INTEGER,               -- Token count
    embedding      VECTOR(1024),          -- Voyage embedding
    git_commit     TEXT,                  -- Source commit
    indexed_at     TIMESTAMP
);

-- Vector similarity index
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
```

## External services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **PostgreSQL + pgvector** | Vector storage and search | `docker-compose.yml` |
| **Voyage AI** | Text embeddings | `VOYAGE_API_KEY` |
| **Anthropic Claude** | LLM generation | `ANTHROPIC_API_KEY` |
| **HuggingFace** | NLI models | Optional `HF_TOKEN` |

## Deployment architecture

### Development

```
┌─────────────────────────────────────────┐
│              Local Machine               │
│  ┌─────────────┐    ┌─────────────────┐ │
│  │   Python    │    │    Docker       │ │
│  │   App       │───▶│   PostgreSQL    │ │
│  └─────────────┘    └─────────────────┘ │
│         │                               │
│         ▼                               │
│  ┌─────────────────────────────────────┐│
│  │         External APIs               ││
│  │  Voyage AI │ Anthropic │ HuggingFace││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### Production (recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  API 1   │    │  API 2   │    │  API 3   │
        │ (FastAPI)│    │ (FastAPI)│    │ (FastAPI)│
        └──────────┘    └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   (pgvector)    │
                    │   Primary       │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Replica       │
                    └─────────────────┘
```

## Performance considerations

### Embedding batching

Embeddings are generated in batches of 128 to minimize API calls:

```python
for i in range(0, len(chunks), 128):
    batch = chunks[i:i + 128]
    embedded = embedder.embed_chunks(batch)
```

### Vector index tuning

The IVFFlat index trades accuracy for speed:

```sql
-- More lists = faster but less accurate
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Connection pooling

AsyncPG connection pool for concurrent queries:

```python
self.pool = await asyncpg.create_pool(
    self.connection_string,
    min_size=5,
    max_size=20
)
```
