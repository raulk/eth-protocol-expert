# Ethereum Protocol Intelligence System

set dotenv-load

# List available recipes
default:
    @just --list

# --- Infrastructure ---

# Start PostgreSQL + pgvector and FalkorDB
up:
    docker compose up -d

# Stop all services
down:
    docker compose down

# Build (or rebuild) container images
build:
    docker compose build

# Rebuild images and restart services
refresh:
    docker compose build
    docker compose up -d

# Reset database (deletes all data)
reset:
    docker compose down -v
    docker compose up -d

# Show service status
status:
    docker compose ps

# Show service logs (optionally for a specific service)
logs service="":
    docker compose logs {{ if service != "" { service } else { "--tail=50" } }}

# --- Querying ---

# Query the system (usage: just query "What is EIP-1559?" mode)
query question mode="agentic":
    uv run python scripts/query_cli.py "{{ question }}" --mode {{ mode }}

# --- Ingestion ---

# Ingest all sources (or a specific one: just ingest eips, just ingest consensus-specs, etc.)
ingest source="all":
    uv run python scripts/ingest_{{ replace(source, "-", "_") }}.py

# List available ingestion sources
ingestions:
    @ls scripts/ingest_*.py | sed 's|scripts/ingest_||;s|\.py||;s|_|-|g' | sort

# --- API ---

# Start the API server (development)
api:
    uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# --- Development ---

# Run tests
test *args="":
    uv run pytest tests/ -v {{ args }}

# Run end-to-end test
test-e2e:
    uv run python scripts/test_e2e.py

# Lint and format
lint:
    uv run ruff check . --fix
    uv run ruff format .

# Check without fixing
check:
    uv run ruff check .
    uv run ruff format . --check
