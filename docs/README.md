# Documentation

Welcome to the Ethereum Protocol Intelligence System documentation.

## Contents

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, setup, and first query |
| [CONCEPTS.md](CONCEPTS.md) | Key concepts: EIPs, RAG, validation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and data flow |
| [QUERIES.md](QUERIES.md) | How to query effectively |
| [API.md](API.md) | REST API reference |

## Quick links

### Getting started

```bash
# Install
uv sync

# Configure
cp .env.example .env
# Edit .env with API keys

# Start database
docker compose up -d

# Ingest EIPs
uv run python scripts/ingest_eips.py

# Query
uv run python scripts/query_cli.py "What is EIP-1559?"
```

### Query modes

| Mode | Command | Use case |
|------|---------|----------|
| Simple | `--mode simple` | Fast exploration |
| Cited | `--mode cited` | Source attribution |
| Validated | `--mode validated` | Fact-checking |

### Key files

| File | Purpose |
|------|---------|
| `scripts/query_cli.py` | Command-line interface |
| `scripts/ingest_eips.py` | EIP ingestion |
| `src/api/main.py` | REST API server |
| `CLAUDE.md` | AI assistant instructions |
| `README.md` | Project overview |
