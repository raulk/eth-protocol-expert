# Ethereum Protocol Intelligence System

An agentic RAG system for Ethereum protocol knowledge. Ingests 20+ data sources (EIPs, ERCs, consensus/execution specs, client codebases, research papers, forum discussions), builds a knowledge graph of relationships, and answers questions with citation-backed, NLI-validated responses.

**Web UI:** http://localhost:3080 (after `docker compose up -d`)

<img width="1472" height="1110" alt="image" src="https://github.com/user-attachments/assets/4439671f-5ddb-4f15-bdcc-46242e3b54c5" />

## Capabilities

**Agentic retrieval.** A ReAct agent reasons about queries, retrieves evidence across multiple hops, reflects on quality, backtracks from dead ends, and synthesizes answers with full reasoning chain visibility.

**Hybrid search.** BM25 full-text search combined with vector similarity (Voyage AI), fused via Reciprocal Rank Fusion. Graph-augmented retrieval follows EIP dependency chains. Adaptive budget allocation scales retrieval effort to query complexity.

**Claim validation.** Responses are decomposed into atomic claims, each verified against source evidence using Natural Language Inference. Claims are classified as supported, weakly supported, or unsupported, with a trustworthiness ratio.

**Knowledge graph.** FalkorDB stores EIP dependencies (REQUIRES, SUPERSEDES), cross-references, citation links, and inferred relationships with confidence scores. Spec-to-implementation linking maps EIPs to Go/Rust client code.

**Structured outputs.** Timelines, argument maps (pro/con analysis), comparison tables, and dependency graph visualizations generated from retrieved evidence.

**Multi-model ensemble.** Cost-aware routing (haiku/sonnet/opus) based on query complexity. Parallel model execution with confidence calibration, circuit breaking for low-evidence queries, and A/B testing.

## Data sources

| Category | Sources |
|----------|---------|
| Standards | EIPs, ERCs, RIPs |
| Specs | Consensus specs, execution specs, execution APIs, beacon APIs, builder specs, DevP2P, Portal Network |
| Forums | ethresear.ch, Ethereum Magicians |
| Research | arXiv papers, Vitalik's research, All Core Devs transcripts |
| Code | go-ethereum, prysm, lighthouse, reth |

## Quick start

```bash
# Install
uv sync

# Start infrastructure (PostgreSQL + pgvector, FalkorDB)
docker compose up -d

# Configure credentials
cp .env.example .env  # then edit with your keys

# Ingest EIPs
uv run python scripts/ingest_eips.py

# Ingest everything
uv run python scripts/ingest_all.py

# Query
uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple
uv run python scripts/query_cli.py "How do blob transactions work?" --mode cited
uv run python scripts/query_cli.py "Is the base fee claim accurate?" --mode validated
uv run python scripts/query_cli.py "How does EIP-1559 interact with EIP-4844?" --mode agentic
```

## Query modes

| Mode | Description |
|------|-------------|
| `simple` | Fast generation with retrieved context |
| `cited` | Inline citations with source attribution |
| `validated` | NLI verification flags unsupported claims |
| `agentic` | ReAct loop with multi-hop reasoning and backtracking |
| `graph` | Dependency-aware retrieval following EIP relationships |

## Architecture

```
src/
├── ingestion/       # 25 loaders for all data sources
├── chunking/        # Fixed, section, forum, paper, code, transcript chunkers
├── embeddings/      # Voyage AI (voyage-4-large), code embeddings (voyage-code-2)
├── storage/         # PostgreSQL + pgvector
├── retrieval/       # Vector, BM25, hybrid, graph-augmented, reranked, staged
├── generation/      # Simple, cited, validated, synthesis generators
├── validation/      # NLI claim verification, claim decomposition
├── evidence/        # Evidence spans, ledgers, support classification
├── concepts/        # Alias tables, query expansion, concept resolution
├── agents/          # ReAct agent, budget enforcement, reflection, backtracking
├── routing/         # Query classification, decomposition, sub-question planning
├── graph/           # EIP dependencies, cross-references, relationship inference
├── structured/      # Timelines, argument maps, comparisons, dependency views
├── ensemble/        # Cost routing, multi-model, confidence calibration, circuit breaker
├── parsing/         # Tree-sitter Go/Rust parsing, code unit extraction
├── dedup/           # MinHash/SimHash near-duplicate detection
├── filters/         # Metadata filtering (status, type, category, date)
└── api/             # FastAPI REST API

app/                 # React + TypeScript frontend
```

## API

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query with configurable mode |
| `/health` | GET | Health check with document/chunk counts |
| `/stats` | GET | System statistics |
| `/eip/{number}` | GET | EIP metadata and chunks |
| `/search` | GET | Vector search |
| `/eip/{number}/dependencies` | GET | Dependency chain |
| `/eip/{number}/dependents` | GET | Reverse dependencies |
| `/eip/{number}/tree` | GET | Dependency tree visualization |
| `/graph/stats` | GET | Graph statistics |
| `/graph/most-depended` | GET | Most depended-upon EIPs |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VOYAGE_API_KEY` | Yes | Voyage AI for embeddings |
| `ANTHROPIC_API_KEY` | Yes | Anthropic for generation |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `DATABASE_URL` | No | Full PostgreSQL connection URL (auto-generated if not set) |
| `HF_TOKEN` | No | HuggingFace token (for gated NLI models) |

## Development

```bash
uv run pytest tests/ -v          # unit tests
uv run python scripts/test_e2e.py # end-to-end test
uv run ruff check . --fix         # lint
uv run ruff format .              # format
```

## License

MIT
