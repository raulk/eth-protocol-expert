# FalkorDB API integration plan

## Problem

FalkorDB graph database is running and graph code is fully implemented (`src/graph/`), but:

1. **Ingestion doesn't populate the graph** - `ingest_eips.py` parses EIPs with relationship data but never calls `EIPGraphBuilder`
2. **API doesn't expose graph features** - Graph-augmented retrieval code exists but isn't wired to endpoints
3. **docker-compose.yml missing FalkorDB** - Container was started manually, not managed

## Current state

| Component | Status |
|-----------|--------|
| FalkorDB container | Running (orphan, started manually) |
| `src/graph/falkordb_store.py` | Complete |
| `src/graph/eip_graph_builder.py` | Complete |
| `src/graph/dependency_traverser.py` | Complete |
| `src/retrieval/graph_augmented.py` | Complete |
| **Ingestion → Graph integration** | **Missing** |
| **API integration** | **Missing** |
| docker-compose.yml | Missing FalkorDB service |

## Root cause

In `scripts/ingest_eips.py` lines 94-99:

```python
for i, loaded_eip in enumerate(loaded_eips):
    parsed = parser.parse(loaded_eip)
    parsed_eips.append(parsed)  # ← Has requires, superseded-by, etc.
    chunks = chunker.chunk_eip(parsed)
    all_chunks.extend(chunks)
```

The `parsed_eips` list contains all relationship data needed for the graph, but it's only used for PostgreSQL storage. `EIPGraphBuilder.build_from_eips(parsed_eips)` is never called.

## Goals

1. **Fix ingestion to populate graph during EIP ingest**
2. Add FalkorDB to docker-compose.yml
3. Initialize graph store in API lifespan
4. Expose graph-augmented retrieval via `/query` endpoint
5. Add dedicated graph endpoints for EIP dependencies

## Implementation phases

### Phase 1: Fix ingestion pipeline (critical)

Update `scripts/ingest_eips.py` to populate graph during ingestion:

```python
# Add imports
from src.graph import FalkorDBStore, EIPGraphBuilder

async def ingest_eips(...):
    # ... existing setup ...

    # After parsing all EIPs (around line 108)
    logger.info("chunking_complete", total_chunks=len(all_chunks), total_eips=len(parsed_eips))

    # NEW: Build EIP graph from parsed data
    logger.info("building_eip_graph")
    graph_store = FalkorDBStore(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", 6379)),
    )
    graph_store.connect()

    try:
        builder = EIPGraphBuilder(graph_store)
        graph_result = builder.build_from_eips(parsed_eips)
        logger.info(
            "graph_built",
            nodes=graph_result.nodes_created,
            relationships=graph_result.relationships_created,
            requires=graph_result.requires_count,
            supersedes=graph_result.supersedes_count,
        )
    finally:
        graph_store.close()

    # ... continue with existing embedding/storage ...
```

Add environment variables to `.env`:
```
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
```

### Phase 2: Docker compose integration

Add FalkorDB service to `docker-compose.yml`:

```yaml
falkordb:
  image: falkordb/falkordb:latest
  container_name: eth-protocol-falkordb
  expose:
    - "6379"
  volumes:
    - falkordb_data:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 5s
    timeout: 3s
    retries: 5
```

Update API service environment:
```yaml
FALKORDB_HOST: falkordb
FALKORDB_PORT: 6379
```

Add to volumes:
```yaml
volumes:
  pgdata:
  falkordb_data:
```

### Phase 3: API lifespan initialization

Update `src/api/main.py`:

```python
import os
from ..graph import FalkorDBStore, DependencyTraverser
from ..retrieval import GraphAugmentedRetriever

# Global instances
graph_store: FalkorDBStore | None = None
dependency_traverser: DependencyTraverser | None = None

# In lifespan():
try:
    graph_store = FalkorDBStore(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", 6379)),
    )
    graph_store.connect()
    dependency_traverser = DependencyTraverser(graph_store)
    logger.info("graph_store_initialized")
except Exception as e:
    logger.warning("graph_store_not_available", error=str(e))
    graph_store = None
```

### Phase 4: Dedicated graph endpoints

Add to `src/api/main.py`:

```python
@app.get("/eip/{eip_number}/dependencies")
async def get_eip_dependencies(eip_number: int, depth: int = 3):
    """Get EIP dependency chain (what this EIP requires)."""
    if not dependency_traverser:
        raise HTTPException(status_code=503, detail="Graph database not available")

    chain = dependency_traverser.get_requires_chain(eip_number, max_depth=depth)
    return {
        "eip": eip_number,
        "dependencies": chain,
        "depth": depth,
    }

@app.get("/eip/{eip_number}/dependents")
async def get_eip_dependents(eip_number: int):
    """Get EIPs that depend on this one."""
    if not dependency_traverser:
        raise HTTPException(status_code=503, detail="Graph database not available")

    dependents = dependency_traverser.get_required_by(eip_number)
    return {
        "eip": eip_number,
        "dependents": dependents,
    }

@app.get("/graph/stats")
async def get_graph_stats():
    """Get graph statistics."""
    if not graph_store:
        raise HTTPException(status_code=503, detail="Graph database not available")

    return {
        "nodes": graph_store.count_nodes(),
        "relationships": graph_store.count_relationships(),
    }
```

### Phase 5: Graph-augmented query mode

Add `graph` mode to `/query` endpoint that uses `GraphAugmentedRetriever`:

- When querying about an EIP, automatically include related EIPs in context
- Return graph context in response (dependencies, dependents)

Update `QueryRequest` model:
```python
mode: str = Field(
    default="cited",
    description="Generation mode: 'simple', 'cited', 'validated', 'agentic', or 'graph'"
)
```

Update `QueryResponse` model:
```python
# Graph fields
related_eips: list[str] | None = None
dependency_chain: list[str] | None = None
```

### Phase 6: Frontend integration

Update `app/src/App.tsx`:

1. Add `graph` to QueryMode type
2. Show dependency chain in results
3. Add clickable EIP links

## Execution order

1. [x] Add FalkorDB to docker-compose.yml with volume
2. [x] Add FALKORDB_HOST/PORT to .env
3. [x] Update ingest_eips.py to build graph during ingestion
4. [x] Re-run ingestion to populate graph
5. [x] Update API with graph store initialization
6. [x] Add graph endpoints (/eip/{id}/dependencies, /graph/stats)
7. [x] Add graph mode to /query endpoint
8. [x] Update frontend with graph mode
9. [x] Test end-to-end

## Environment variables

Add to `.env`:
```
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
```

For docker-compose, API service gets:
```
FALKORDB_HOST=falkordb
FALKORDB_PORT=6379
```

## Testing

```bash
# Rebuild with graph population
uv run python scripts/ingest_eips.py --limit 100

# Check graph stats
curl http://localhost:3001/api/graph/stats

# Test dependencies endpoint
curl http://localhost:3001/api/eip/1559/dependencies

# Test dependents endpoint
curl http://localhost:3001/api/eip/1559/dependents

# Test graph-augmented query
curl -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What EIPs does EIP-4844 depend on?", "mode": "graph"}'
```

## Notes

- Existing FalkorDB container has been running 12 days with anonymous volume - data may be stale or empty
- After fixing ingestion, need to re-run to populate graph
- Graph population is fast (seconds) since it just extracts frontmatter relationships
- Graph retrieval adds minimal latency (~10ms) for dependency lookups
