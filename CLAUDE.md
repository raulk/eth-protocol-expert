# Ethereum Protocol Intelligence System

RAG system for Ethereum protocol documentation with citation validation.

## Project overview

This is a 14-phase RAG (Retrieval-Augmented Generation) system that answers questions about Ethereum Improvement Proposals (EIPs). All phases (0-13) are implemented.

**Core pipeline:** Ingest EIPs → Chunk → Embed → Store → Retrieve → Generate → Validate

## Key commands

```bash
# Start database
docker compose up -d

# Ingest all EIPs
uv run python scripts/ingest_eips.py

# Sync and ingest ethresear.ch forum
uv run python scripts/sync_ethresearch.py --stats-only      # Show stats
uv run python scripts/sync_ethresearch.py --max-topics 1000 # Sync to cache
uv run python scripts/sync_ethresearch.py --incremental     # Incremental sync
uv run python scripts/ingest_ethresearch.py --skip-existing # Ingest from cache

# Query the system
uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple
uv run python scripts/query_cli.py "What is EIP-1559?" --mode cited
uv run python scripts/query_cli.py "What is EIP-1559?" --mode validated
uv run python scripts/query_cli.py "How does EIP-1559 interact with EIP-4844?" --mode agentic

# Run tests
uv run pytest tests/ -v
uv run python scripts/test_e2e.py

# Lint and format
uv run ruff check . --fix
uv run ruff format .
```

## Architecture

```
src/
├── ingestion/      # EIP loading, parsing, extended corpus loaders
│   ├── cache.py           # Raw content cache for API sources
│   ├── discourse_client.py    # Discourse forum API client
│   └── ethresearch_loader.py  # ethresear.ch sync/load
├── chunking/       # Fixed, section-aware, and code chunking
├── embeddings/     # Voyage AI, local BGE, code embeddings
├── storage/        # PostgreSQL + pgvector
├── retrieval/      # Vector similarity search
├── generation/     # Simple, cited, validated, synthesis generators
├── validation/     # NLI claim verification, decomposition
├── evidence/       # Evidence spans and ledgers
├── concepts/       # Alias tables, query expansion (Phase 7)
├── agents/         # ReAct agents, budget enforcement (Phase 8)
├── structured/     # Timelines, argument maps (Phase 9)
├── graph/          # EIP dependencies, cross-references (Phases 4, 11, 13)
├── ensemble/       # Cost routing, multi-model (Phase 12)
├── parsing/        # Tree-sitter code analysis (Phase 13)
└── api/            # FastAPI endpoints

data/
├── eips/           # Cloned EIPs repository (git-based cache)
└── cache/          # Raw content cache for API sources
    └── ethresearch/    # Forum topics JSON cache
```

## Module conventions

### Imports

Use absolute imports from `src`:
```python
from src.chunking import FixedChunker, SectionChunker
from src.embeddings import VoyageEmbedder
from src.storage import PgVectorStore
from src.retrieval import SimpleRetriever
from src.generation import SimpleGenerator, CitedGenerator, ValidatedGenerator
```

### Async patterns

- Database operations are async (use `await`)
- Embedding is sync (VoyageEmbedder.embed_chunks returns directly)
- Generation uses `asyncio.to_thread` for Claude API calls

```python
# Correct pattern
store = PgVectorStore()
await store.connect()
chunks = chunker.chunk_eip(parsed)  # sync
embedded = embedder.embed_chunks(chunks)  # sync
await store.store_embedded_chunks(embedded)  # async
```

### Dataclasses

Key dataclasses to know:

| Class | Module | Purpose |
|-------|--------|---------|
| `LoadedEIP` | ingestion | Raw EIP from filesystem |
| `ParsedEIP` | ingestion | Parsed with frontmatter |
| `LoadedForumTopic` | ingestion | Forum topic with posts |
| `CacheEntry` | ingestion | Cached content metadata |
| `Chunk` | chunking | Text chunk with metadata |
| `EmbeddedChunk` | embeddings | Chunk with vector |
| `SearchResult` | retrieval | Chunk with similarity score |
| `RetrievalResult` | retrieval | Collection of search results |
| `GenerationResult` | generation | Response with metadata |
| `EvidenceSpan` | evidence | Immutable source reference |
| `Claim` | evidence | Extracted claim from response |

## Database

PostgreSQL with pgvector extension. Schema:

- `documents` - EIP metadata (number, title, status, type, category)
- `chunks` - Text chunks with embeddings (1024-dim vectors)

```bash
# Connection
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/eth_protocol

# Reset database
docker compose down -v && docker compose up -d
```

## Environment variables

Required in `.env`:
```
VOYAGE_API_KEY=...       # Voyage AI for embeddings
ANTHROPIC_API_KEY=...    # Anthropic for generation
POSTGRES_PASSWORD=...    # Database password
```

## Testing

```bash
# Unit tests
uv run pytest tests/ -v

# End-to-end (requires database)
uv run python scripts/test_e2e.py

# Full RAG pipeline
uv run python scripts/test_full_rag.py
```

## Common tasks

### Add a new EIP source

1. Create loader in `src/ingestion/` (follow `eip_loader.py` pattern)
2. Add parser if needed
3. Export from `src/ingestion/__init__.py`

### Add a new generator mode

1. Create in `src/generation/` (extend base pattern)
2. Add to `src/generation/__init__.py`
3. Add mode to `scripts/query_cli.py`

### Add a new chunking strategy

1. Create in `src/chunking/`
2. Implement `chunk_eip(parsed: ParsedEIP) -> list[Chunk]`
3. Export from `src/chunking/__init__.py`

## Important notes

### NLI validation

Default model is `facebook/bart-large-mnli` (public, no auth).
For better quality, use `microsoft/deberta-v3-large-mnli` (requires HF login).

### Embedding dimensions

- Voyage AI (voyage-4-large): 1024 dimensions (default, supports 256/512/1024/2048)
- Local BGE: 1024 dimensions
- Code embeddings (voyage-code-2): 1536 dimensions

### API field names

- `SearchResult.similarity` (not `score`)
- `GenerationResult.response` (not `answer`)
- `RetrievalResult.results` (list of SearchResult)

## File locations

| Purpose | Location |
|---------|----------|
| Main CLI | `scripts/query_cli.py` |
| EIP ingestion | `scripts/ingest_eips.py` |
| Forum sync | `scripts/sync_ethresearch.py` |
| Forum ingest | `scripts/ingest_ethresearch.py` |
| Raw content cache | `src/ingestion/cache.py` |
| Database schema | `src/storage/pg_vector_store.py` |
| API server | `src/api/main.py` |
| Implementation plan | `archive/plans-and-roadmaps` branch |
| Full spec | `archive/plans-and-roadmaps` branch |

## Phase summary

| Phase | Module | Key classes |
|-------|--------|-------------|
| 0-2 | generation | SimpleGenerator, CitedGenerator, ValidatedGenerator |
| 3 | retrieval | HybridRetriever (BM25 + vector) |
| 4 | graph | EIPGraphBuilder, DependencyTraverser |
| 5 | routing | QueryDecomposer |
| 6 | ingestion | MagiciansLoader, EthresearchLoader, RawContentCache |
| 7 | concepts | AliasTable, QueryExpander, ConceptResolver |
| 8 | agents | ReactAgent, BudgetEnforcer, Backtracker |
| 9 | structured | TimelineBuilder, ArgumentMapper, ComparisonBuilder |
| 10 | ingestion | ACDTranscriptLoader, ArxivFetcher, PDFExtractor |
| 11 | graph | RelationshipInferrer, ConfidenceScorer |
| 12 | ensemble | CostRouter, MultiModelRunner |
| 13 | parsing | TreeSitterParser, SpecImplLinker |

## Forum ingestion

The forum pipeline uses a two-stage sync/ingest pattern:

```bash
# Stage 1: Sync to cache (downloads from API)
uv run python scripts/sync_ethresearch.py --max-topics 1000

# Stage 2: Ingest from cache (chunks, embeds, stores)
uv run python scripts/ingest_ethresearch.py --skip-existing
```

Key features:
- **Incremental sync**: Uses `bumped_at` to only fetch modified topics
- **Smart staleness**: Compares `posts_count` and `bumped_at` vs cached values
- **Pipelined ingest**: Producer (chunking) runs concurrently with consumer (embedding)
- **Batched embeddings**: Collects chunks across topics for efficient API calls
