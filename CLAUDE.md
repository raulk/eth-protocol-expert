# Ethereum Protocol Intelligence System

Agentic RAG system for Ethereum protocol knowledge with citation validation, knowledge graphs, and multi-model ensemble.

## Key commands

```bash
docker compose up -d                                          # start infrastructure
uv run python scripts/ingest_eips.py                          # ingest EIPs
uv run python scripts/ingest_all.py                           # ingest all sources
uv run python scripts/sync_ethresearch.py --max-topics 1000   # sync forum to cache
uv run python scripts/ingest_ethresearch.py --skip-existing   # ingest forum from cache
uv run python scripts/query_cli.py "question" --mode agentic  # query (simple|cited|validated|agentic|graph)
uv run pytest tests/ -v                                       # tests
uv run ruff check . --fix && uv run ruff format .             # lint + format
```

## Architecture

```
src/
├── ingestion/      # 25 loaders: EIPs, ERCs, RIPs, specs, forums, papers, client code
├── chunking/       # Fixed, section, forum, paper, code, transcript chunkers
├── embeddings/     # Voyage AI (voyage-4-large, 1024-dim), code (voyage-code-2, 1536-dim)
├── storage/        # PostgreSQL + pgvector
├── retrieval/      # Vector, BM25, hybrid (RRF), graph-augmented, reranked, staged
├── generation/     # Simple, cited, validated, synthesis generators (Claude)
├── validation/     # NLI claim verification (bart-large-mnli / deberta-v3-large-mnli)
├── evidence/       # Immutable evidence spans, ledgers, support classification
├── concepts/       # Alias tables, query expansion, concept resolution
├── agents/         # ReAct agent, budget enforcement, reflection, backtracking
├── routing/        # Query classification, decomposition, sub-question planning
├── graph/          # FalkorDB: EIP deps, cross-refs, relationship inference, spec-impl linking
├── structured/     # Timelines, argument maps, comparisons, dependency views
├── ensemble/       # Cost routing, multi-model, confidence calibration, circuit breaker, A/B testing
├── parsing/        # Tree-sitter Go/Rust parsing, code unit extraction
├── dedup/          # MinHash/SimHash near-duplicate detection
├── filters/        # Metadata filtering (status, type, category, date)
└── api/            # FastAPI REST API

app/                # React + TypeScript frontend (Vite, TanStack Query)
```

## Module conventions

### Imports

Absolute imports from `src`:
```python
from src.chunking import FixedChunker, SectionChunker
from src.embeddings import VoyageEmbedder
from src.storage import PgVectorStore
from src.retrieval import SimpleRetriever
from src.generation import SimpleGenerator, CitedGenerator, ValidatedGenerator
```

### Async patterns

Database operations are async. Embedding is sync. Generation uses `asyncio.to_thread` for Claude API calls.

```python
store = PgVectorStore()
await store.connect()
chunks = chunker.chunk_eip(parsed)          # sync
embedded = embedder.embed_chunks(chunks)    # sync
await store.store_embedded_chunks(embedded) # async
```

## Key dataclasses

| Class | Module | Purpose |
|-------|--------|---------|
| `ParsedEIP` | ingestion | Parsed EIP with frontmatter + sections |
| `LoadedForumTopic` | ingestion | Forum topic with posts |
| `Chunk` | chunking | Text chunk with metadata |
| `EmbeddedChunk` | embeddings | Chunk with vector |
| `SearchResult` | retrieval | Chunk with similarity score |
| `RetrievalResult` | retrieval | Collection of search results |
| `GenerationResult` | generation | Response with metadata |
| `EvidenceSpan` | evidence | Immutable source reference |
| `AgentState` | agents | Tracks thoughts, retrievals, budget |
| `AgentResult` | agents | Answer + reasoning chain + metadata |
| `RoutingDecision` | ensemble | Model tier + estimated cost |

## API field names

- `SearchResult.similarity` (not `score`)
- `GenerationResult.response` (not `answer`)
- `RetrievalResult.results` (list of SearchResult)

## Infrastructure

PostgreSQL with pgvector (vector storage) and FalkorDB (knowledge graph). Both bound to 127.0.0.1.

```bash
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/eth_protocol
docker compose down -v && docker compose up -d  # reset
```

## Environment variables

Required in `.env`:
```
VOYAGE_API_KEY=...       # Voyage AI for embeddings
ANTHROPIC_API_KEY=...    # Anthropic for generation
POSTGRES_PASSWORD=...    # Database password
```

## Common tasks

### Add a new data source

1. Create loader in `src/ingestion/` (follow existing loader patterns)
2. Add chunker in `src/chunking/` if content type needs specialized chunking
3. Create ingestion script in `scripts/`
4. Export from `src/ingestion/__init__.py`

### Add a new query mode

1. Create generator in `src/generation/`
2. Export from `src/generation/__init__.py`
3. Add mode to `scripts/query_cli.py` and `src/api/main.py`

## Module summary

| Area | Module | Key classes |
|------|--------|-------------|
| Ingestion | ingestion | 25 loaders, IngestionOrchestrator, RawContentCache |
| Chunking | chunking | FixedChunker, SectionChunker, ForumChunker, CodeChunker, PaperChunker, TranscriptChunker |
| Retrieval | retrieval | SimpleRetriever, BM25Retriever, HybridRetriever, GraphAugmentedRetriever, CodeRetriever |
| Generation | generation | SimpleGenerator, CitedGenerator, ValidatedGenerator, SynthesisGenerator |
| Validation | validation | NLIValidator, ClaimDecomposer, CitationEnforcer, ResponseVerifier |
| Agents | agents | ReactAgent, BudgetEnforcer, Reflector, Backtracker, QueryAnalyzer |
| Routing | routing | QueryClassifier, QueryDecomposer |
| Graph | graph | EIPGraphBuilder, DependencyTraverser, RelationshipInferrer, SpecImplLinker |
| Structured | structured | TimelineBuilder, ArgumentMapper, ComparisonBuilder, DependencyViewBuilder |
| Ensemble | ensemble | CostRouter, MultiModelRunner, ConfidenceCalibrator, CircuitBreaker |
| Parsing | parsing | TreeSitterParser, CodeUnitExtractor |
| Dedup | dedup | DedupService, MinHasher, SimHasher |
