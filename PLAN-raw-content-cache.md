# Raw content cache implementation plan

## Problem statement

When re-ingesting content with different parameters (chunking strategy, embedding model, etc.), API-based sources (arXiv, ethresearch, magicians) require re-fetching from remote servers. This is:

1. **Slow** - Network latency, rate limits, API pagination
2. **Wasteful** - Re-downloading unchanged content
3. **Fragile** - API availability, rate limit errors, 403s

Git-based sources (EIPs, consensus-specs, execution-specs, pm) already cache naturally via local clones.

## Current state

| Source | Caching | Location | Re-ingest cost |
|--------|---------|----------|----------------|
| EIPs | ✅ git clone | `data/eips/` | Fast |
| Consensus specs | ✅ git clone | `data/consensus-specs/` | Fast |
| Execution specs | ✅ git clone | `data/execution-specs/` | Fast |
| ACD transcripts | ✅ git clone | `data/pm/` | Fast |
| arXiv papers | ❌ None | N/A | Slow (API + PDF downloads) |
| Ethresearch | ❌ None | N/A | Slow (API pagination) |
| Magicians | ❌ None | N/A | Slow (API pagination) |

## Proposed solution

Add a `RawContentCache` that stores fetched content locally before processing.

### Directory structure

```
data/
├── eips/                    # git clone (unchanged)
├── consensus-specs/         # git clone (unchanged)
├── execution-specs/         # git clone (unchanged)
├── pm/                      # git clone (unchanged)
└── cache/                   # NEW: raw content cache
    ├── arxiv/
    │   ├── index.json       # metadata index
    │   ├── 2301.00001/
    │   │   ├── meta.json    # paper metadata
    │   │   └── paper.pdf    # downloaded PDF
    │   └── 2301.00002/
    │       ├── meta.json
    │       └── paper.pdf
    ├── ethresearch/
    │   ├── index.json       # metadata index
    │   └── topics/
    │       ├── 12345.json   # full topic with posts
    │       └── 12346.json
    └── magicians/
        ├── index.json       # metadata index
        └── topics/
            ├── 67890.json   # full topic with posts
            └── 67891.json
```

### Cache metadata schema

Each cached item includes metadata for staleness detection:

```python
@dataclass
class CacheEntry:
    """Metadata for a cached item."""

    source: str              # "arxiv", "ethresearch", "magicians"
    item_id: str             # "2301.00001", "topic-12345"
    fetched_at: datetime     # When we downloaded it

    # Source-specific timestamps for staleness
    updated_at: datetime | None    # Last modified at source
    etag: str | None               # HTTP ETag if available

    # Content location
    content_path: Path       # Relative path to content file
    content_hash: str        # SHA256 of content for integrity

    # Size tracking
    content_bytes: int       # Size in bytes
```

### Index file format

Each source has an `index.json` for fast lookups:

```json
{
  "source": "arxiv",
  "last_sync": "2026-01-20T14:30:00Z",
  "entries": {
    "2301.00001": {
      "fetched_at": "2026-01-20T14:30:00Z",
      "updated_at": "2023-01-15T00:00:00Z",
      "content_path": "2301.00001/paper.pdf",
      "content_hash": "sha256:abc123...",
      "content_bytes": 1234567,
      "meta": {
        "title": "Paper Title",
        "authors": ["Author 1", "Author 2"],
        "categories": ["cs.CR", "cs.DC"]
      }
    }
  }
}
```

---

## Implementation phases

### Phase 1: Core cache infrastructure

**File:** `src/ingestion/cache.py`

```python
"""Raw content cache for API-sourced documents."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Metadata for a cached item."""

    source: str
    item_id: str
    fetched_at: str  # ISO format
    updated_at: str | None
    etag: str | None
    content_path: str
    content_hash: str
    content_bytes: int
    meta: dict


class RawContentCache:
    """Local cache for raw content fetched from APIs.

    Provides cache-first loading: check cache, fetch only if missing/stale.
    """

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self._indexes: dict[str, dict] = {}

    def _get_source_dir(self, source: str) -> Path:
        """Get directory for a source."""
        return self.cache_dir / source

    def _get_index_path(self, source: str) -> Path:
        """Get path to source index file."""
        return self._get_source_dir(source) / "index.json"

    def _load_index(self, source: str) -> dict:
        """Load or create index for a source."""
        if source in self._indexes:
            return self._indexes[source]

        index_path = self._get_index_path(source)
        if index_path.exists():
            with open(index_path) as f:
                self._indexes[source] = json.load(f)
        else:
            self._indexes[source] = {
                "source": source,
                "last_sync": None,
                "entries": {}
            }

        return self._indexes[source]

    def _save_index(self, source: str) -> None:
        """Save index to disk."""
        index = self._indexes.get(source)
        if not index:
            return

        index_path = self._get_index_path(source)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def has(self, source: str, item_id: str) -> bool:
        """Check if item is cached."""
        index = self._load_index(source)
        return item_id in index["entries"]

    def get(self, source: str, item_id: str) -> CacheEntry | None:
        """Get cache entry metadata."""
        index = self._load_index(source)
        entry_data = index["entries"].get(item_id)
        if not entry_data:
            return None
        return CacheEntry(**entry_data)

    def get_content_path(self, source: str, item_id: str) -> Path | None:
        """Get full path to cached content file."""
        entry = self.get(source, item_id)
        if not entry:
            return None
        return self._get_source_dir(source) / entry.content_path

    def get_content(self, source: str, item_id: str) -> bytes | None:
        """Read cached content."""
        path = self.get_content_path(source, item_id)
        if not path or not path.exists():
            return None
        return path.read_bytes()

    def get_content_text(self, source: str, item_id: str) -> str | None:
        """Read cached content as text."""
        content = self.get_content(source, item_id)
        if content is None:
            return None
        return content.decode("utf-8")

    def put(
        self,
        source: str,
        item_id: str,
        content: bytes,
        meta: dict,
        updated_at: datetime | None = None,
        etag: str | None = None,
        content_subpath: str | None = None,
    ) -> CacheEntry:
        """Store content in cache.

        Args:
            source: Source identifier (e.g., "arxiv")
            item_id: Unique item ID within source
            content: Raw content bytes
            meta: Source-specific metadata dict
            updated_at: Last modified time at source
            etag: HTTP ETag for conditional requests
            content_subpath: Override default content path

        Returns:
            CacheEntry for the stored item
        """
        source_dir = self._get_source_dir(source)

        # Default path structure: {source}/{item_id}/content.{ext}
        if content_subpath:
            content_path = content_subpath
        else:
            # Detect extension from content or default to .json
            content_path = f"{item_id}/content.bin"

        full_path = source_dir / content_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

        content_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"

        entry = CacheEntry(
            source=source,
            item_id=item_id,
            fetched_at=datetime.utcnow().isoformat(),
            updated_at=updated_at.isoformat() if updated_at else None,
            etag=etag,
            content_path=content_path,
            content_hash=content_hash,
            content_bytes=len(content),
            meta=meta,
        )

        index = self._load_index(source)
        index["entries"][item_id] = asdict(entry)
        index["last_sync"] = datetime.utcnow().isoformat()
        self._save_index(source)

        logger.debug("cached_content", source=source, item_id=item_id, bytes=len(content))
        return entry

    def is_stale(
        self,
        source: str,
        item_id: str,
        max_age_hours: float = 24 * 7,  # 1 week default
    ) -> bool:
        """Check if cached item is stale.

        Args:
            source: Source identifier
            item_id: Item ID
            max_age_hours: Maximum cache age in hours

        Returns:
            True if missing or older than max_age_hours
        """
        entry = self.get(source, item_id)
        if not entry:
            return True

        fetched_at = datetime.fromisoformat(entry.fetched_at)
        age_hours = (datetime.utcnow() - fetched_at).total_seconds() / 3600
        return age_hours > max_age_hours

    def list_entries(self, source: str) -> list[CacheEntry]:
        """List all cached entries for a source."""
        index = self._load_index(source)
        return [CacheEntry(**data) for data in index["entries"].values()]

    def stats(self, source: str) -> dict:
        """Get cache statistics for a source."""
        index = self._load_index(source)
        entries = index["entries"]

        total_bytes = sum(e.get("content_bytes", 0) for e in entries.values())

        return {
            "source": source,
            "entry_count": len(entries),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "last_sync": index.get("last_sync"),
        }
```

**Export from `src/ingestion/__init__.py`:**
```python
from .cache import CacheEntry, RawContentCache
```

---

### Phase 2: Cached arXiv loader

Modify `ArxivFetcher` to use cache for PDF downloads.

**File:** `src/ingestion/arxiv_fetcher.py` (modifications)

```python
# Add to imports
from .cache import RawContentCache

class ArxivFetcher:
    def __init__(
        self,
        timeout: float = 30.0,
        cache: RawContentCache | None = None,
    ):
        self.timeout = timeout
        self.cache = cache or RawContentCache()
        self._client: httpx.Client | None = None

    def fetch_pdf_cached(self, paper: ArxivPaper) -> Path | None:
        """Fetch PDF with caching.

        Returns path to cached PDF file.
        """
        if not paper.pdf_url:
            return None

        item_id = paper.arxiv_id

        # Check cache first
        cached_path = self.cache.get_content_path("arxiv", item_id)
        if cached_path and cached_path.exists():
            logger.debug("using_cached_pdf", arxiv_id=item_id)
            return cached_path

        # Fetch and cache
        logger.info("downloading_pdf", arxiv_id=item_id, url=paper.pdf_url)
        try:
            response = self._get_client().get(paper.pdf_url, follow_redirects=True)
            response.raise_for_status()
            pdf_content = response.content
        except httpx.HTTPError as e:
            logger.error("pdf_download_failed", arxiv_id=item_id, error=str(e))
            return None

        # Store in cache
        meta = {
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat() if paper.published else None,
            "abstract": paper.abstract,
        }

        self.cache.put(
            source="arxiv",
            item_id=item_id,
            content=pdf_content,
            meta=meta,
            updated_at=paper.updated,
            content_subpath=f"{item_id}/paper.pdf",
        )

        return self.cache.get_content_path("arxiv", item_id)

    def search_ethereum_papers_cached(
        self,
        max_results: int = 100,
        cache_metadata: bool = True,
    ) -> list[ArxivPaper]:
        """Search with optional metadata caching.

        Even if we cache search results, we should refresh periodically
        since new papers are published.
        """
        papers = self.search_ethereum_papers(max_results=max_results)

        if cache_metadata:
            # Store paper metadata (not PDFs) for quick re-access
            for paper in papers:
                if not self.cache.has("arxiv", paper.arxiv_id):
                    meta = {
                        "title": paper.title,
                        "authors": paper.authors,
                        "categories": paper.categories,
                        "published": paper.published.isoformat() if paper.published else None,
                        "abstract": paper.abstract,
                        "pdf_url": paper.pdf_url,
                    }
                    # Store metadata only (no PDF yet)
                    self.cache.put(
                        source="arxiv",
                        item_id=paper.arxiv_id,
                        content=json.dumps(meta).encode(),
                        meta=meta,
                        updated_at=paper.updated,
                        content_subpath=f"{paper.arxiv_id}/meta.json",
                    )

        return papers
```

---

### Phase 3: Cached Discourse loader

Modify `EthresearchLoader` and `MagiciansLoader` to cache topics.

**File:** `src/ingestion/ethresearch_loader.py` (modifications)

```python
# Add to imports
import json
from .cache import RawContentCache

class EthresearchLoader:
    def __init__(
        self,
        base_url: str = ETHRESEARCH_BASE_URL,
        rate_limit_delay: float = 1.0,
        cache: RawContentCache | None = None,
    ):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.source = "ethresearch"
        self.cache = cache or RawContentCache()

    def _topic_to_cache_dict(self, topic: LoadedForumTopic) -> dict:
        """Convert topic to cacheable dict."""
        return {
            "topic_id": topic.topic_id,
            "title": topic.title,
            "slug": topic.slug,
            "category": topic.category,
            "tags": topic.tags,
            "posts_count": topic.posts_count,
            "created_at": topic.created_at.isoformat(),
            "last_posted_at": topic.last_posted_at.isoformat() if topic.last_posted_at else None,
            "posts": [
                {
                    "post_id": p.post_id,
                    "post_number": p.post_number,
                    "username": p.username,
                    "content": p.content,
                    "reply_to_post_number": p.reply_to_post_number,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                }
                for p in topic.posts
            ],
        }

    def _cache_dict_to_topic(self, data: dict) -> LoadedForumTopic:
        """Reconstruct topic from cached dict."""
        posts = [
            LoadedForumPost(
                source=self.source,
                topic_id=data["topic_id"],
                post_id=p["post_id"],
                post_number=p["post_number"],
                username=p["username"],
                content=p["content"],
                reply_to_post_number=p["reply_to_post_number"],
                created_at=datetime.fromisoformat(p["created_at"]),
                updated_at=datetime.fromisoformat(p["updated_at"]),
                topic_title=data["title"],
                category=data["category"],
                tags=data["tags"],
            )
            for p in data["posts"]
        ]

        return LoadedForumTopic(
            source=self.source,
            topic_id=data["topic_id"],
            title=data["title"],
            slug=data["slug"],
            category=data["category"],
            tags=data["tags"],
            posts_count=data["posts_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_posted_at=datetime.fromisoformat(data["last_posted_at"]) if data["last_posted_at"] else None,
            posts=posts,
        )

    async def load_topic_with_posts_cached(
        self,
        topic_id: int,
        max_age_hours: float = 24 * 7,
    ) -> LoadedForumTopic | None:
        """Load topic with caching."""
        item_id = f"topic-{topic_id}"

        # Check cache
        if not self.cache.is_stale(self.source, item_id, max_age_hours):
            cached_data = self.cache.get_content_text(self.source, item_id)
            if cached_data:
                logger.debug("using_cached_topic", topic_id=topic_id)
                return self._cache_dict_to_topic(json.loads(cached_data))

        # Fetch from API
        topic = await self.load_topic_with_posts(topic_id)
        if not topic:
            return None

        # Cache it
        topic_dict = self._topic_to_cache_dict(topic)
        self.cache.put(
            source=self.source,
            item_id=item_id,
            content=json.dumps(topic_dict, indent=2).encode(),
            meta={"title": topic.title, "posts_count": len(topic.posts)},
            updated_at=topic.last_posted_at,
            content_subpath=f"topics/{topic_id}.json",
        )

        return topic

    async def iter_topics_with_posts_cached(
        self,
        max_topics: int = 100,
        max_age_hours: float = 24 * 7,
    ) -> AsyncIterator[LoadedForumTopic]:
        """Iterate through topics with caching."""
        async with self._make_client() as client:
            topic_count = 0
            async for topic in client.iter_all_topics():
                if topic_count >= max_topics:
                    break

                loaded = await self.load_topic_with_posts_cached(
                    topic.topic_id,
                    max_age_hours=max_age_hours,
                )
                if loaded:
                    yield loaded
                    topic_count += 1
```

Apply same pattern to `MagiciansLoader`.

---

### Phase 4: Update ingestion scripts

**File:** `scripts/ingest_arxiv.py` (modifications)

```python
from src.ingestion import ArxivFetcher, PDFExtractor, QualityScorer
from src.ingestion.cache import RawContentCache

async def ingest_arxiv(
    max_papers: int = 300,
    use_cache: bool = True,
    max_cache_age_hours: float = 24 * 7,
) -> None:
    """Ingest arXiv papers with optional caching."""
    cache = RawContentCache() if use_cache else None
    fetcher = ArxivFetcher(cache=cache)
    extractor = PDFExtractor()
    # ... rest of ingestion

    for paper in papers:
        if use_cache:
            # Use cached PDF if available
            pdf_path = fetcher.fetch_pdf_cached(paper)
            if pdf_path:
                pdf_content = extractor.extract(pdf_path)
            else:
                continue
        else:
            # Original behavior: download to temp file
            pdf_content = extractor.extract_from_url(paper.pdf_url)
```

**File:** `scripts/ingest_ethresearch.py` (modifications)

```python
async def ingest_ethresearch(
    max_topics: int = 1000,
    use_cache: bool = True,
    max_cache_age_hours: float = 24 * 7,
) -> None:
    """Ingest ethresear.ch topics with optional caching."""
    cache = RawContentCache() if use_cache else None
    loader = EthresearchLoader(cache=cache)

    if use_cache:
        async for topic in loader.iter_topics_with_posts_cached(
            max_topics=max_topics,
            max_age_hours=max_cache_age_hours,
        ):
            # ... process topic
    else:
        async for topic in loader.iter_topics_with_posts(max_topics=max_topics):
            # ... process topic
```

---

### Phase 5: Cache management CLI

**File:** `scripts/cache_cli.py`

```python
#!/usr/bin/env python3
"""Cache management CLI.

Usage:
    uv run python scripts/cache_cli.py stats
    uv run python scripts/cache_cli.py stats --source arxiv
    uv run python scripts/cache_cli.py list --source ethresearch
    uv run python scripts/cache_cli.py clear --source arxiv --older-than 30d
    uv run python scripts/cache_cli.py verify --source arxiv
"""

import argparse
from src.ingestion.cache import RawContentCache


def cmd_stats(args):
    """Show cache statistics."""
    cache = RawContentCache()
    sources = [args.source] if args.source else ["arxiv", "ethresearch", "magicians"]

    for source in sources:
        stats = cache.stats(source)
        print(f"\n{source}:")
        print(f"  Entries: {stats['entry_count']}")
        print(f"  Size: {stats['total_mb']} MB")
        print(f"  Last sync: {stats['last_sync']}")


def cmd_list(args):
    """List cached entries."""
    cache = RawContentCache()
    entries = cache.list_entries(args.source)

    for entry in entries[:args.limit]:
        print(f"{entry.item_id}: {entry.meta.get('title', 'N/A')[:60]}")
        print(f"  Fetched: {entry.fetched_at}")
        print(f"  Size: {entry.content_bytes} bytes")


def cmd_clear(args):
    """Clear cache entries."""
    # Implementation: delete entries older than threshold
    pass


def cmd_verify(args):
    """Verify cache integrity (check hashes)."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Cache management CLI")
    subparsers = parser.add_subparsers(dest="command")

    # stats
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--source", choices=["arxiv", "ethresearch", "magicians"])

    # list
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--source", required=True)
    list_parser.add_argument("--limit", type=int, default=50)

    # clear
    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument("--source", required=True)
    clear_parser.add_argument("--older-than", help="e.g., 30d, 1w")

    # verify
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--source", required=True)

    args = parser.parse_args()

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
```

---

### Phase 6: Update .gitignore

Add cache directory (large files shouldn't be committed):

```
# Data directories
data/eips/
data/code/
data/papers/
data/transcripts/
data/cache/           # NEW: raw content cache
data/consensus-specs/
data/execution-specs/
data/pm/
```

---

## Migration plan

1. **No migration needed** - Cache is additive
2. **First run with cache** - Will fetch all content and populate cache
3. **Subsequent runs** - Will use cached content
4. **Force refresh** - Use `--no-cache` or `--max-cache-age 0`

---

## CLI interface changes

```bash
# Default: use cache
uv run python scripts/ingest_arxiv.py --max-papers 300

# Disable cache (force re-fetch)
uv run python scripts/ingest_arxiv.py --max-papers 300 --no-cache

# Custom cache age (fetch if older than 1 day)
uv run python scripts/ingest_arxiv.py --max-papers 300 --max-cache-age 24

# Cache management
uv run python scripts/cache_cli.py stats
uv run python scripts/cache_cli.py list --source arxiv
```

---

## Expected outcomes

| Scenario | Before | After |
|----------|--------|-------|
| Re-ingest with new chunking | ~30 min (re-fetch all) | ~2 min (read from cache) |
| Re-ingest with new embeddings | ~30 min | ~5 min (embed cached content) |
| Partial failure recovery | Start over | Resume from cache |
| Test new pipeline | Full fetch each time | Instant from cache |

---

## Validation queries

After implementation, verify:

```python
# Check cache populated
cache = RawContentCache()
assert cache.stats("arxiv")["entry_count"] > 0
assert cache.stats("ethresearch")["entry_count"] > 0

# Check re-ingest uses cache
# Run ingest twice, second should be much faster
# Check logs for "using_cached_*" messages
```

---

## Files to create/modify

| File | Action |
|------|--------|
| `src/ingestion/cache.py` | Create |
| `src/ingestion/__init__.py` | Add exports |
| `src/ingestion/arxiv_fetcher.py` | Modify |
| `src/ingestion/ethresearch_loader.py` | Modify |
| `src/ingestion/magicians_loader.py` | Modify |
| `scripts/ingest_arxiv.py` | Modify |
| `scripts/ingest_ethresearch.py` | Modify |
| `scripts/ingest_magicians.py` | Modify |
| `scripts/cache_cli.py` | Create |
| `.gitignore` | Modify |

---

## Implementation order

1. [ ] Create `src/ingestion/cache.py` with `RawContentCache` class
2. [ ] Export from `src/ingestion/__init__.py`
3. [ ] Add `fetch_pdf_cached` to `ArxivFetcher`
4. [ ] Add caching methods to `EthresearchLoader`
5. [ ] Add caching methods to `MagiciansLoader`
6. [ ] Update `scripts/ingest_arxiv.py`
7. [ ] Update `scripts/ingest_ethresearch.py`
8. [ ] Update `scripts/ingest_magicians.py`
9. [ ] Create `scripts/cache_cli.py`
10. [ ] Update `.gitignore`
11. [ ] Test full ingestion cycle with caching
12. [ ] Test re-ingestion uses cache
