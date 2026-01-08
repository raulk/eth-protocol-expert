# Ethereum Protocol Intelligence System

A RAG (Retrieval-Augmented Generation) system for Ethereum protocol documentation, progressing from simple Q&A to validated, citation-backed answers.

## Phases implemented

### Phase 0: Hello World RAG ✅
- Basic EIP ingestion and parsing
- Fixed-size chunking
- Vector embeddings (Voyage AI)
- Top-k retrieval
- Simple generation (Claude)

### Phase 1: Trustworthy Citations ✅
- Section-aware chunking (respects EIP structure)
- Atomic code blocks (never split)
- Evidence spans with document/section tracking
- Inline citation formatting

### Phase 2: Citation Validation ✅
- NLI-based claim verification (DeBERTa)
- Claim decomposition into atomic facts
- Support level classification (STRONG/PARTIAL/WEAK/NONE/CONTRADICTION)
- Validation reports and warnings

## Quick start

### 1. Set up environment

```bash
# Clone the repo
cd eth-protocol-expert

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# - VOYAGE_API_KEY: Get from https://www.voyageai.com/
# - ANTHROPIC_API_KEY: Get from https://console.anthropic.com/
```

### 3. Start database

```bash
sudo docker compose up -d
```

### 4. Ingest EIPs

```bash
python scripts/ingest_eips.py
```

This will:
- Clone the ethereum/EIPs repository
- Parse all EIP markdown files
- Create section-aware chunks
- Generate embeddings
- Store everything in PostgreSQL

### 5. Query the system

**CLI:**
```bash
# Simple mode (Phase 0)
python scripts/query_cli.py "What is EIP-4844?" --mode simple

# With citations (Phase 1)
python scripts/query_cli.py "What is EIP-4844?" --mode cited

# With validation (Phase 2) - requires NLI model
python scripts/query_cli.py "What is EIP-4844?" --mode validated
```

**API:**
```bash
# Start the API server
uvicorn src.api.main:app --reload

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is EIP-4844?", "mode": "cited"}'
```

## Project structure

```
eth-protocol-expert/
├── src/
│   ├── ingestion/          # EIP loading and parsing
│   │   ├── eip_loader.py   # Clone/load EIPs from GitHub
│   │   └── eip_parser.py   # Parse markdown, extract frontmatter
│   ├── chunking/           # Text chunking strategies
│   │   ├── fixed_chunker.py    # Phase 0: Simple fixed-size
│   │   └── section_chunker.py  # Phase 1: Section-aware
│   ├── embeddings/         # Vector embeddings
│   │   └── voyage_embedder.py  # Voyage AI embeddings
│   ├── storage/            # Database storage
│   │   └── pg_vector_store.py  # PostgreSQL + pgvector
│   ├── retrieval/          # Search and retrieval
│   │   └── simple_retriever.py # Vector similarity search
│   ├── evidence/           # Citation tracking
│   │   ├── evidence_span.py    # Immutable evidence references
│   │   └── evidence_ledger.py  # Claim-to-evidence mapping
│   ├── validation/         # Citation validation
│   │   ├── nli_validator.py    # NLI-based verification
│   │   └── claim_decomposer.py # Atomic fact extraction
│   ├── generation/         # Response generation
│   │   ├── simple_generator.py    # Phase 0
│   │   ├── cited_generator.py     # Phase 1
│   │   └── validated_generator.py # Phase 2
│   └── api/                # FastAPI application
│       └── main.py
├── scripts/
│   ├── ingest_eips.py      # Ingestion script
│   └── query_cli.py        # CLI query tool
├── tests/                  # Test suite
├── data/eips/              # Cloned EIPs repository
├── docker-compose.yml      # PostgreSQL + pgvector
└── pyproject.toml          # Dependencies
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/query` | POST | Query with configurable mode |
| `/eip/{number}` | GET | Get EIP metadata and chunks |
| `/search` | GET | Search for relevant chunks |

## Query modes

| Mode | Description | Features |
|------|-------------|----------|
| `simple` | Phase 0 | Fast, basic RAG |
| `cited` | Phase 1 | Source citations |
| `validated` | Phase 2 | NLI verification |

## Running tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_eip_parser.py -v
```

## Requirements

- Python 3.11+
- Docker (for PostgreSQL)
- API keys:
  - Voyage AI (embeddings)
  - Anthropic (generation)

## Next phases (see IMPLEMENTATION_PLAN.md)

- Phase 3: Hybrid search (BM25 + vectors + reranking)
- Phase 4: EIP dependency graph
- Phase 5: Query decomposition
- Phase 6+: Forums, agentic retrieval, structured outputs, and more
