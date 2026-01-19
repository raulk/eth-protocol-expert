# Ethereum Protocol Intelligence System

A production-grade RAG (Retrieval-Augmented Generation) system for Ethereum protocol documentation. Query EIPs, understand protocol mechanics, and get citation-backed, validated answers.

## Features

- **881 EIPs indexed** with 8,500+ semantic chunks
- **Three query modes**: simple, cited, and NLI-validated
- **Section-aware chunking** that respects EIP document structure
- **Citation tracking** with evidence ledgers and source attribution
- **NLI validation** to flag unsupported or contradicted claims
- **Concept resolution** with alias tables for Ethereum terminology
- **Multi-model ensemble** with cost-aware routing
- **Graph-based relationships** for EIP dependencies and cross-references

## Quick start

### 1. Prerequisites

- Python 3.11+
- Docker (for PostgreSQL with pgvector)
- API keys:
  - [Voyage AI](https://www.voyageai.com/) for embeddings
  - [Anthropic](https://console.anthropic.com/) for generation

### 2. Installation

```bash
# Clone and enter directory
git clone <repo-url>
cd eth-protocol-expert

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### 3. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `VOYAGE_API_KEY` | Voyage AI API key for embeddings |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `POSTGRES_PASSWORD` | Password for PostgreSQL database |

### 4. Start the database

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Verify it's running
docker compose ps
```

### 5. Ingest EIPs

```bash
# Ingest all EIPs (clones ethereum/EIPs repo, ~880 documents)
uv run python scripts/ingest_eips.py

# Or with options
uv run python scripts/ingest_eips.py --use-section-chunking --batch-size 50

# For testing with fewer EIPs
uv run python scripts/ingest_eips.py --limit 50
```

### 6. Query the system

```bash
# Simple mode - fast, basic RAG
uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple

# Cited mode - with source attribution
uv run python scripts/query_cli.py "How do blob transactions work?" --mode cited

# Validated mode - with NLI verification (flags unsupported claims)
uv run python scripts/query_cli.py "What is the base fee?" --mode validated
```

## Query modes

| Mode | Description | Use case |
|------|-------------|----------|
| `simple` | Fast generation with retrieved context | Quick answers, exploration |
| `cited` | Adds inline citations and source tracking | Research, documentation |
| `validated` | NLI-based claim verification | High-stakes queries, fact-checking |

### Example output (validated mode)

```
Based on the provided context, EIP-1559 introduced a base fee mechanism...

The base fee is dynamic - it adjusts based on block utilization. ⚠️ [WEAKLY SUPPORTED]

------------------------------------------------------------
VALIDATION SUMMARY:
  Total claims: 5
  Supported: 3
  Weak: 1
  Unsupported: 1
  Support ratio: 60.0%
  Trustworthy: No
```

## Architecture

```
eth-protocol-expert/
├── src/
│   ├── ingestion/           # Data loading and parsing
│   │   ├── eip_loader.py        # Clone/load EIPs from GitHub
│   │   ├── eip_parser.py        # Parse markdown, extract frontmatter
│   │   ├── acd_transcript_loader.py  # AllCoreDevs call transcripts
│   │   ├── arxiv_fetcher.py     # Academic paper fetching
│   │   ├── git_code_loader.py   # Ethereum client code loading
│   │   └── pdf_extractor.py     # PDF document extraction
│   │
│   ├── chunking/            # Text chunking strategies
│   │   ├── fixed_chunker.py     # Simple fixed-size chunks
│   │   ├── section_chunker.py   # Section-aware (respects headers)
│   │   └── code_chunker.py      # Function-level code chunking
│   │
│   ├── embeddings/          # Vector embeddings
│   │   ├── voyage_embedder.py   # Voyage AI (voyage-3)
│   │   ├── local_embedder.py    # Local BGE model
│   │   └── code_embedder.py     # Code-specific (voyage-code-2)
│   │
│   ├── storage/             # Database layer
│   │   └── pg_vector_store.py   # PostgreSQL + pgvector
│   │
│   ├── retrieval/           # Search and retrieval
│   │   └── simple_retriever.py  # Vector similarity search
│   │
│   ├── generation/          # Response generation
│   │   ├── simple_generator.py      # Basic RAG generation
│   │   ├── cited_generator.py       # With citations
│   │   ├── validated_generator.py   # With NLI validation
│   │   └── synthesis_generator.py   # Multi-source synthesis
│   │
│   ├── validation/          # Citation validation
│   │   ├── nli_validator.py     # NLI-based verification
│   │   └── claim_decomposer.py  # Atomic fact extraction
│   │
│   ├── evidence/            # Citation tracking
│   │   ├── evidence_span.py     # Immutable evidence references
│   │   └── evidence_ledger.py   # Claim-to-evidence mapping
│   │
│   ├── concepts/            # Concept resolution (Phase 7)
│   │   ├── alias_table.py       # Ethereum terminology aliases
│   │   ├── query_expander.py    # Query term expansion
│   │   └── concept_resolver.py  # Canonical term resolution
│   │
│   ├── agents/              # Agentic retrieval (Phase 8)
│   │   ├── react_agent.py       # ReAct-style agent
│   │   ├── budget_enforcer.py   # Resource limits
│   │   ├── backtrack.py         # Dead-end detection
│   │   └── reflection.py        # Quality assessment
│   │
│   ├── structured/          # Structured outputs (Phase 9)
│   │   ├── timeline_builder.py      # Temporal event ordering
│   │   ├── argument_mapper.py       # Pro/con analysis
│   │   ├── comparison_table.py      # Feature comparisons
│   │   └── dependency_view.py       # Dependency visualization
│   │
│   ├── graph/               # Knowledge graph (Phases 4, 11, 13)
│   │   ├── eip_graph_builder.py     # EIP dependency graph
│   │   ├── cross_reference.py       # Cross-document links
│   │   ├── citation_graph.py        # Academic citations
│   │   ├── relationship_inferrer.py # Semantic inference
│   │   ├── confidence_scorer.py     # Relationship confidence
│   │   └── spec_impl_linker.py      # EIP-to-code mapping
│   │
│   ├── ensemble/            # Multi-model ensemble (Phase 12)
│   │   ├── cost_router.py       # Model selection by cost/quality
│   │   ├── multi_model_runner.py    # Parallel model execution
│   │   └── conditional_trigger.py   # Ensemble activation
│   │
│   ├── parsing/             # Code analysis (Phase 13)
│   │   ├── treesitter_parser.py     # Go/Rust parsing
│   │   └── code_unit_extractor.py   # Function extraction
│   │
│   └── api/                 # REST API
│       └── main.py              # FastAPI application
│
├── scripts/
│   ├── ingest_eips.py       # EIP ingestion pipeline
│   ├── query_cli.py         # Command-line query tool
│   ├── test_e2e.py          # End-to-end test
│   └── test_full_rag.py     # Full RAG pipeline test
│
├── data/eips/               # Cloned EIPs repository
├── docker-compose.yml       # PostgreSQL + pgvector
└── pyproject.toml           # Dependencies and config
```

## Implemented phases

| Phase | Name | Description | Status |
|-------|------|-------------|--------|
| 0 | Hello World RAG | Basic ingestion, chunking, embeddings, retrieval, generation | ✅ |
| 1 | Trustworthy Citations | Section-aware chunking, evidence spans, inline citations | ✅ |
| 2 | Citation Validation | NLI verification, claim decomposition, support classification | ✅ |
| 3 | Hybrid Search | BM25 + vector search, reranking | ✅ |
| 4 | EIP Dependency Graph | Graph database, dependency traversal | ✅ |
| 5 | Query Decomposition | Multi-hop question handling | ✅ |
| 6 | Forum Integration | Ethereum Magicians, ethresear.ch | ✅ |
| 7 | Concept Resolution | Alias tables, query expansion | ✅ |
| 8 | Agentic Retrieval | ReAct agents, budget enforcement, backtracking | ✅ |
| 9 | Structured Outputs | Timelines, argument maps, comparisons | ✅ |
| 10 | Expanded Corpus | ACD transcripts, arXiv papers, PDFs | ✅ |
| 11 | Inferred Relationships | Semantic inference, confidence scoring | ✅ |
| 12 | Multi-Model Ensemble | Cost routing, parallel generation | ✅ |
| 13 | Client Codebase Analysis | Tree-sitter parsing, EIP-to-code linking | ✅ |

## API reference

### Start the API server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics (document/chunk counts) |
| `/query` | POST | Query with configurable mode |
| `/eip/{number}` | GET | Get EIP metadata and chunks |
| `/search` | GET | Search for relevant chunks |

### Query endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the base fee in EIP-1559?",
    "mode": "cited",
    "top_k": 5
  }'
```

## CLI options

```bash
uv run python scripts/query_cli.py --help

Options:
  --mode [simple|cited|validated]  Query mode (default: simple)
  --top-k INTEGER                  Number of chunks to retrieve (default: 5)
  --no-sources                     Hide source citations
  --local                          Use local embeddings (no API)
```

## Development

### Run tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test
uv run pytest tests/test_eip_parser.py -v

# End-to-end test
uv run python scripts/test_e2e.py
```

### Code quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check . --fix
```

### Database management

```bash
# Start database
docker compose up -d

# Stop database
docker compose down

# Reset database (delete all data)
docker compose down -v
docker compose up -d
```

## Configuration

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOYAGE_API_KEY` | Yes | - | Voyage AI API key |
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `POSTGRES_PASSWORD` | Yes | - | Database password |
| `DATABASE_URL` | No | Auto-generated | Full PostgreSQL URL |
| `HF_TOKEN` | No | - | HuggingFace token (for gated models) |

### NLI model options

The validated mode uses NLI (Natural Language Inference) for claim verification:

| Model | Auth required | Quality | Speed |
|-------|---------------|---------|-------|
| `facebook/bart-large-mnli` (default) | No | Good | Fast |
| `microsoft/deberta-v3-large-mnli` | Yes (HF token) | Best | Slower |

To use DeBERTa:
```bash
# Login to HuggingFace
huggingface-cli login

# Set token in environment
export HF_TOKEN=your_token
```

## Troubleshooting

### Database connection errors

```bash
# Check if PostgreSQL is running
docker compose ps

# Check logs
docker compose logs postgres

# Restart
docker compose restart
```

### Embedding API errors

- Verify `VOYAGE_API_KEY` is set correctly
- Check Voyage AI dashboard for rate limits
- Use `--local` flag for offline embeddings

### NLI validation errors

If you see `401 Unauthorized` for HuggingFace:
- The default model (`facebook/bart-large-mnli`) doesn't require auth
- For `deberta-v3-large-mnli`, run `huggingface-cli login`

### Slow queries

- Reduce `--top-k` for faster retrieval
- Use `--mode simple` instead of `validated`
- Consider local embeddings for development

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full technical specification.
