# Ralph Loop: Implement raw content cache and ingest ethresearch

## Goal

Implement a raw content caching layer for API-sourced documents and perform a full ethresearch ingestion with the cache enabled.

## Success criteria

All of the following must be true before outputting the completion promise:

1. **Cache infrastructure exists:**
   - `src/ingestion/cache.py` contains `RawContentCache` class with `has()`, `get()`, `put()`, `is_stale()`, `get_content_path()`, `get_content_text()`, `list_entries()`, `stats()` methods
   - `CacheEntry` dataclass exists with required fields
   - Cache exports added to `src/ingestion/__init__.py`

2. **EthresearchLoader has caching:**
   - `src/ingestion/ethresearch_loader.py` modified to accept `cache` parameter
   - `load_topic_with_posts_cached()` method exists
   - `iter_topics_with_posts_cached()` method exists
   - Serialization/deserialization methods for topics exist

3. **Ingestion script updated:**
   - `scripts/ingest_ethresearch.py` uses cache by default
   - `--no-cache` flag disables caching
   - `--max-cache-age` flag controls staleness threshold

4. **Cache CLI exists:**
   - `scripts/cache_cli.py` with `stats`, `list` subcommands working

5. **Full ethresearch ingestion completes:**
   - Run: `uv run python scripts/ingest_ethresearch.py --max-topics 500`
   - Verify cache populated: `uv run python scripts/cache_cli.py stats --source ethresearch`
   - Cache should show > 0 entries

6. **Code passes quality checks:**
   - `uv run ruff check src/ingestion/cache.py`
   - `uv run ruff check src/ingestion/ethresearch_loader.py`
   - `uv run ruff check scripts/ingest_ethresearch.py`
   - `uv run ruff check scripts/cache_cli.py`

7. **`.gitignore` updated:**
   - `data/cache/` is in `.gitignore`

## Implementation plan

Follow the phases in `PLAN-raw-content-cache.md`:

### Phase 1: Core cache infrastructure
Create `src/ingestion/cache.py` with `RawContentCache` and `CacheEntry`.

### Phase 2: Skip arXiv (focus on ethresearch first)

### Phase 3: Cached Discourse loader
Modify `src/ingestion/ethresearch_loader.py` to add caching methods.

### Phase 4: Update ingestion script
Modify `scripts/ingest_ethresearch.py` to use cache.

### Phase 5: Cache management CLI
Create `scripts/cache_cli.py` with basic functionality.

### Phase 6: Update .gitignore
Add `data/cache/` exclusion.

### Phase 7: Run full ingestion
Execute the ethresearch ingestion and verify cache population.

## Constraints

- Use absolute imports from `src`
- Follow Python 3.12+ syntax (type aliases, union syntax `X | None`)
- No docstrings for self-describing functions
- Use `structlog` for logging
- Use `httpx` for HTTP requests
- Async methods where appropriate (Discourse API is async)
- Cache directory: `data/cache/`
- Cache structure: `data/cache/{source}/topics/{topic_id}.json`

## Key files to reference

- `PLAN-raw-content-cache.md` - Full implementation details
- `src/ingestion/ethresearch_loader.py` - Current loader implementation
- `src/ingestion/discourse_client.py` - Underlying Discourse API client
- `scripts/ingest_ethresearch.py` - Current ingestion script

## Iteration checklist

Each iteration, check:

1. What files have been created/modified?
2. What's the next incomplete step?
3. Are there any errors to fix?
4. Can we run a test to verify progress?

## Completion promise

When ALL success criteria are met (cache working, ethresearch ingested, > 0 entries in cache stats), output:

```
<promise>RAW CONTENT CACHE IMPLEMENTED AND ETHRESEARCH INGESTED</promise>
```

Do NOT output the promise until ALL criteria are verified.
