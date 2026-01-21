# Getting started

This guide will help you set up the Ethereum Protocol Intelligence System and run your first query in under 10 minutes.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **Docker** installed and running
- **API keys** for:
  - [Voyage AI](https://www.voyageai.com/) (free tier available)
  - [Anthropic](https://console.anthropic.com/) (Claude API)

## Step 1: Clone and install

```bash
# Clone the repository
git clone <repo-url>
cd eth-protocol-expert

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Step 2: Configure environment

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Required API keys
VOYAGE_API_KEY=your_voyage_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database password (choose a secure password)
POSTGRES_PASSWORD=your_secure_password_here

# Optional: Full database URL (auto-generated if not set)
DATABASE_URL=postgresql://postgres:your_secure_password_here@localhost:5432/eth_protocol
```

## Step 3: Start the database

The system uses PostgreSQL with the pgvector extension for vector similarity search.

```bash
# Start PostgreSQL in the background
docker compose up -d

# Verify it's running
docker compose ps
```

You should see the `eth-protocol-db` container running.

## Step 4: Ingest EIPs

This step clones the official Ethereum EIPs repository, parses all documents, generates embeddings, and stores them in the database.

```bash
# Full ingestion (~880 EIPs, takes 2-5 minutes)
uv run python scripts/ingest_eips.py

# Or ingest a smaller subset for testing
uv run python scripts/ingest_eips.py --limit 50
```

You'll see progress output:

```
starting_ingestion            data_dir=data/eips section_chunking=True
cloning_or_updating_repo
repo_ready                    commit=d154dfa2
loading_eips
loaded_eips                   count=881
parsing_and_chunking_all_eips
...
ingestion_complete            documents=881 chunks=8522
```

## Step 5: Run your first query

```bash
# Simple query
uv run python scripts/query_cli.py "What is EIP-1559?"
```

Example output:

```
RESPONSE (Simple Mode - Phase 0)
============================================================
EIP-1559 is a significant Ethereum Improvement Proposal that reformed
the transaction fee mechanism. It introduced a base fee that is burned
(removed from circulation) and dynamically adjusts based on network
congestion...

------------------------------------------------------------
Tokens: 1256 input, 268 output
```

## Step 6: Try different query modes

### Cited mode (with source attribution)

```bash
uv run python scripts/query_cli.py "How do blob transactions work?" --mode cited
```

This adds inline citations like `[EIP-4844, Abstract]` and lists sources.

### Validated mode (with claim verification)

```bash
uv run python scripts/query_cli.py "What is the base fee?" --mode validated
```

This verifies claims using NLI and flags unsupported statements:

```
The base fee adjusts dynamically. ⚠️ [WEAKLY SUPPORTED]

VALIDATION SUMMARY:
  Total claims: 5
  Supported: 3
  Weak: 1
  Unsupported: 1
```

## Step 7: Start the API server (optional)

For programmatic access:

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Test with curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is EIP-4844?", "mode": "cited"}'
```

## Step 8: Ingest forum discussions (optional)

The system can also ingest discussions from ethresear.ch (Ethereum Research forum).

### Sync forum topics to local cache

```bash
# Show statistics (forum total vs cached)
uv run python scripts/sync_ethresearch.py --stats-only

# Sync latest 1000 topics
uv run python scripts/sync_ethresearch.py --max-topics 1000

# Incremental sync (only topics modified since last sync)
uv run python scripts/sync_ethresearch.py --incremental
```

### Ingest cached topics into database

```bash
# Ingest from cache (skip already ingested)
uv run python scripts/ingest_ethresearch.py --skip-existing

# With larger embedding batches (faster)
uv run python scripts/ingest_ethresearch.py --skip-existing --batch-size 256
```

The sync/ingest separation allows you to:
- Re-chunk or re-embed without re-downloading
- Resume after interruption
- Update only changed content with incremental sync

## Next steps

- Read [CONCEPTS.md](CONCEPTS.md) to understand EIPs and RAG terminology
- Read [QUERIES.md](QUERIES.md) for advanced query techniques
- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand system internals
- Read [API.md](API.md) for full API documentation

## Common issues

### Database connection failed

```bash
# Check if PostgreSQL is running
docker compose ps

# Restart if needed
docker compose restart

# Check logs
docker compose logs postgres
```

### API key errors

- Verify keys are set in `.env`
- Ensure no extra whitespace around keys
- Check that `.env` is in the project root

### Slow first query

The first validated query downloads the NLI model (~1.5GB). Subsequent queries are faster.

### Out of memory

For systems with limited RAM, use local embeddings:

```bash
uv run python scripts/ingest_eips.py --local
uv run python scripts/query_cli.py "What is EIP-1559?" --local
```
