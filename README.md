# Ethereum Protocol Intelligence System

An agentic RAG system for Ethereum protocol knowledge. Ingests 20+ data sources (EIPs, ERCs, consensus/execution specs, client codebases, research papers, forum discussions), builds a knowledge graph of relationships, and answers questions with citation-backed, NLI-validated responses.

**Web UI:** http://localhost:3080 (after `docker compose up -d`)

<img width="1472" height="1110" alt="image" src="https://github.com/user-attachments/assets/4439671f-5ddb-4f15-bdcc-46242e3b54c5" />

## Capabilities

**Agentic retrieval.** A ReAct agent reasons about queries, retrieves evidence across multiple hops, reflects on quality, backtracks from dead ends, and synthesizes answers with full reasoning chain visibility.

**Hybrid search.** BM25 full-text search combined with vector similarity (Voyage AI), fused via Reciprocal Rank Fusion. Graph-augmented retrieval follows EIP dependency chains. Adaptive budget allocation scales retrieval effort to query complexity.

**Claim validation.** Responses can be decomposed into atomic claims, each verified against source evidence using Natural Language Inference. Claims are classified as supported, weakly supported, or unsupported. (Currently disabled at runtime; the NLI model dependencies are not installed by default.)

**Knowledge graph.** FalkorDB stores EIP dependencies (REQUIRES, SUPERSEDES), cross-references, citation links, and inferred relationships with confidence scores. Spec-to-implementation linking maps EIPs to Go/Rust client code.

**Structured outputs.** Timelines, argument maps (pro/con analysis), comparison tables, and dependency graph visualizations generated from retrieved evidence.

**Multi-provider LLM.** Generation routes through a unified completion layer (litellm) supporting Anthropic, Google Gemini, and OpenRouter (hundreds of models). The frontend exposes a searchable model picker with per-token pricing. Cost-aware routing, confidence calibration, and circuit breaking for low-evidence queries.

## Data sources

| Category | Sources |
|----------|---------|
| Standards | EIPs, ERCs, RIPs |
| Specs | Consensus specs, execution specs, execution APIs, beacon APIs, builder specs, DevP2P, Portal Network |
| Forums | ethresear.ch, Ethereum Magicians |
| Research | arXiv papers, Vitalik's research, All Core Devs transcripts |
| Code | go-ethereum, prysm, lighthouse, reth |

## Ingestion pipeline

Every source follows the same pattern: load raw content, chunk it into pieces sized for embedding, send chunks to Voyage AI to get vectors, then store the vectors in PostgreSQL (pgvector). Voyage is stateless; it returns vectors and retains nothing. At query time, the user's question is embedded (one Voyage call) and matched against stored vectors locally.

The chunker and embedding model vary by content type:

| Source | Chunker | Embedding model | Preprocessing |
|--------|---------|-----------------|---------------|
| EIPs, ERCs, RIPs | SectionChunker | voyage-4-large | Git clone, markdown parsing, frontmatter extraction |
| Consensus specs | Custom markdown splitter | voyage-4-large | Regex section splitting, preserves code blocks atomically |
| Execution specs | Custom Python splitter | voyage-4-large | Token-aware fixed chunking with overlap (512 tokens, 64 overlap) |
| Execution APIs | SectionChunker | voyage-4-large | Markdown section extraction from ethereum/execution-apis |
| Beacon APIs | SectionChunker | voyage-4-large | OpenAPI endpoint definitions converted to markdown |
| Builder specs, DevP2P, Portal specs | SectionChunker | voyage-4-large | Markdown section extraction from respective repos |
| ethresear.ch | ForumChunker | voyage-4-large | Discourse API sync to local cache (incremental), pipelined chunking+embedding |
| Ethereum Magicians | ForumChunker | voyage-4-large | Live Discourse API queries, speaker-aware post chunking |
| arXiv papers | PaperChunker | voyage-4-large | PDF extraction (PyMuPDF), quality scoring (rejects < 0.5), section-aware splits preserving equations |
| Vitalik's research | SectionChunker | voyage-4-large | Markdown and Python content, split by file type |
| ACD transcripts | TranscriptChunker | voyage-4-large | Speaker-aware chunking from ethereum/pm repo |
| Client codebases | CodeChunker | voyage-code-3 | Tree-sitter parsing (Go/Rust), function-level extraction, structured text prep (file path, function name, dependencies) |

All text sources target 512 tokens per chunk. Batches sent to Voyage are capped at 100K tokens (below the 120K API limit) to avoid rejected requests.

## Quick start

```bash
# Install
uv sync

# Start infrastructure (PostgreSQL + pgvector, FalkorDB)
just up

# Configure credentials
cp .env.example .env  # then edit with your keys

# List available ingestion sources
just ingestions

# Ingest a specific source
just ingest eips

# Ingest everything
just ingest all

# Sync ethresear.ch forum (incremental)
uv run python scripts/sync_ethresearch.py

# Query
just query "What is EIP-1559?" simple
just query "How do blob transactions work?" cited
just query "Is the base fee claim accurate?" validated
just query "How does EIP-1559 interact with EIP-4844?" agentic
```

## Query modes

| Mode | Description |
|------|-------------|
| `simple` | Fast generation with retrieved context |
| `cited` | Inline citations with source attribution |
| `validated` | NLI verification flags unsupported claims (requires torch/transformers; disabled by default) |
| `agentic` | ReAct loop with multi-hop reasoning and backtracking |
| `graph` | Dependency-aware retrieval following EIP relationships |

## Architecture

```
src/
├── ingestion/       # 25 loaders for all data sources
├── chunking/        # Fixed, section, forum, paper, code, transcript chunkers
├── embeddings/      # Voyage AI (voyage-4-large), code embeddings (voyage-code-3)
├── storage/         # PostgreSQL + pgvector
├── retrieval/       # Vector, BM25, hybrid, graph-augmented, reranked, staged
├── generation/      # Simple, cited, validated, synthesis generators (multi-provider via completion.py)
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
just api
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query with configurable mode, model, and max tokens |
| `/models` | GET | Available models with pricing (Anthropic, Gemini, OpenRouter) |
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
| `ANTHROPIC_API_KEY` | Yes | Anthropic for generation (default provider) |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `GEMINI_API_KEY` | No | Google Gemini models |
| `OPENROUTER_API_KEY` | No | OpenRouter (hundreds of models) |
| `DATABASE_URL` | No | Full PostgreSQL connection URL (auto-generated if not set) |

## Development

```bash
just test             # unit tests
just test-e2e         # end-to-end test
just lint             # lint + format
just check            # check without fixing
```

## License

MIT
