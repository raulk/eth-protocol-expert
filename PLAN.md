# Autonomous corpus ingestion spec for eth-protocol-expert

## Mission

Build a fully operational Ethereum Protocol Expert RAG system by ingesting all relevant knowledge sources. The system must answer complex Ethereum protocol questions with factually correct, cited answers.

## Success criteria

1. All data sources ingested into PostgreSQL with pgvector
2. System passes validation with these test queries:
   - "What are the different approaches that have been suggested over time for encrypted mempools?"
   - "What are the upstream dependencies of ZK EVMs?"
   - "Why was RISC-V chosen over Wasm?"
3. Each query returns relevant sources from multiple document types
4. Generated answers include proper citations

---

## Execution loop

**For each phase below:**
1. Implement the required changes
2. Run the validation command
3. If validation fails, debug and fix
4. Move to next phase only when current phase passes

---

## Phase 1: Verify prerequisites

### 1.1 Check database is running

```bash
docker compose ps
```

If not running:
```bash
docker compose up -d
```

Wait for healthy status:
```bash
docker compose ps | grep healthy
```

### 1.2 Check dependencies installed

```bash
uv sync
```

### 1.3 Verify existing EIP ingestion works

```bash
uv run python scripts/ingest_eips.py --limit 10
```

Expected: Completes without error, shows "Stored X chunks"

### 1.4 Verify query works

```bash
uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple
```

Expected: Returns a coherent answer about EIP-1559

**Phase 1 validation:** All 4 commands succeed.

---

## Phase 2: Enhance database schema

### 2.1 Read current storage implementation

Read file: `src/storage/pg_vector_store.py`

Understand:
- Current `store_document()` method signature
- Current `store_embedded_chunks()` method signature
- Database connection pattern

### 2.2 Add store_generic_document method

Edit `src/storage/pg_vector_store.py` to add this method to the `PgVectorStore` class:

```python
async def store_generic_document(
    self,
    document_id: str,
    document_type: str,
    title: str,
    source: str,
    raw_content: str,
    metadata: dict[str, Any] | None = None,
    author: str | None = None,
    git_commit: str | None = None,
) -> None:
    """Store a generic document (forum post, transcript, paper, spec).

    Args:
        document_id: Unique identifier (e.g., "ethresearch-topic-1234")
        document_type: One of: forum_topic, acd_transcript, arxiv_paper, consensus_spec, execution_spec
        title: Document title
        source: Source identifier (e.g., "ethresearch", "ethereum/pm")
        raw_content: Full text content
        metadata: Additional JSON metadata
        author: Author name(s)
        git_commit: Git commit hash if from a repo
    """
    if not self._pool:
        raise RuntimeError("Not connected to database")

    async with self._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                eip_number, title, status, type, category, author, requires, raw_content, git_commit,
                document_type, source, metadata
            ) VALUES (
                NULL, $1, NULL, NULL, NULL, $2, NULL, $3, $4,
                $5, $6, $7
            )
            ON CONFLICT (eip_number) DO NOTHING
            """,
            title,
            author,
            raw_content,
            git_commit,
            document_type,
            source,
            json.dumps(metadata or {}),
        )
```

### 2.3 Update schema initialization

Find the `_init_schema` method in `src/storage/pg_vector_store.py` and update the documents table creation to include new columns:

```python
# In _init_schema method, update CREATE TABLE documents to:
await conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        eip_number INTEGER UNIQUE,
        title TEXT,
        status TEXT,
        type TEXT,
        category TEXT,
        author TEXT,
        requires INTEGER[],
        raw_content TEXT,
        git_commit TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        document_type VARCHAR(64) DEFAULT 'eip',
        source VARCHAR(128),
        metadata JSONB DEFAULT '{}'
    )
""")
```

Also add migration for existing databases:

```python
# Add after table creation
await conn.execute("""
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type VARCHAR(64) DEFAULT 'eip';
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS source VARCHAR(128);
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
""")
```

### 2.4 Add document_id to chunks table

The chunks need to reference generic documents by document_id string, not just eip_number. Update chunks table:

```python
# In _init_schema, update chunks table to add document_id column:
await conn.execute("""
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_id VARCHAR(256);
""")
```

### 2.5 Update store_embedded_chunks for generic documents

Add parameter to `store_embedded_chunks`:

```python
async def store_embedded_chunks(
    self,
    chunks: list[EmbeddedChunk],
    git_commit: str | None = None,
    document_id: str | None = None,  # Add this parameter
) -> None:
```

In the INSERT statement, use document_id when provided.

**Phase 2 validation:**
```bash
uv run python -c "
import asyncio
from src.storage import PgVectorStore

async def test():
    store = PgVectorStore()
    await store.connect()
    print('Connected to database')
    await store.close()

asyncio.run(test())
"
```

---

## Phase 3: Create consensus spec loader

### 3.1 Create the loader file

Create file: `src/ingestion/consensus_spec_loader.py`

```python
"""Loader for ethereum/consensus-specs repository."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import re

import structlog

logger = structlog.get_logger()


@dataclass
class ConsensusSpec:
    """A parsed consensus layer specification document."""

    name: str
    fork: str
    title: str
    content: str
    file_path: Path


class ConsensusSpecLoader:
    """Load consensus layer specs from ethereum/consensus-specs repository."""

    REPO_URL = "https://github.com/ethereum/consensus-specs.git"
    FORKS = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra"]

    def __init__(self, repo_path: str | Path = "data/consensus-specs") -> None:
        self.repo_path = Path(repo_path)

    def clone_or_update(self) -> str:
        """Clone or update the consensus-specs repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_consensus_specs", url=self.REPO_URL)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "100", self.REPO_URL, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_consensus_specs")
            subprocess.run(
                ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_all_specs(self) -> list[ConsensusSpec]:
        """Load all specification files from the repository."""
        specs: list[ConsensusSpec] = []

        specs_dir = self.repo_path / "specs"
        if not specs_dir.exists():
            logger.warning("specs_dir_not_found", path=str(specs_dir))
            return specs

        for fork in self.FORKS:
            fork_dir = specs_dir / fork
            if not fork_dir.exists():
                continue

            for md_file in fork_dir.glob("*.md"):
                spec = self._parse_spec_file(md_file, fork)
                if spec:
                    specs.append(spec)
                    logger.debug("loaded_spec", name=spec.name, fork=fork)

        logger.info("loaded_consensus_specs", count=len(specs))
        return specs

    def _parse_spec_file(self, file_path: Path, fork: str) -> ConsensusSpec | None:
        """Parse a single specification markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_spec", path=str(file_path), error=str(e))
            return None

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        name = file_path.stem  # e.g., "beacon-chain", "fork-choice"

        return ConsensusSpec(
            name=name,
            fork=fork,
            title=title,
            content=content,
            file_path=file_path,
        )
```

### 3.2 Export the loader

Edit `src/ingestion/__init__.py` to add:

```python
from src.ingestion.consensus_spec_loader import ConsensusSpec, ConsensusSpecLoader
```

And add to `__all__`:
```python
"ConsensusSpec",
"ConsensusSpecLoader",
```

**Phase 3 validation:**
```bash
uv run python -c "
from src.ingestion import ConsensusSpecLoader

loader = ConsensusSpecLoader()
commit = loader.clone_or_update()
print(f'Commit: {commit}')
specs = loader.load_all_specs()
print(f'Loaded {len(specs)} consensus specs')
for spec in specs[:3]:
    print(f'  - {spec.fork}/{spec.name}: {spec.title[:50]}')
"
```

---

## Phase 4: Create execution spec loader

### 4.1 Create the loader file

Create file: `src/ingestion/execution_spec_loader.py`

```python
"""Loader for ethereum/execution-specs repository."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

import structlog

logger = structlog.get_logger()


@dataclass
class ExecutionSpec:
    """A parsed execution layer specification file."""

    module_path: str
    fork: str
    content: str
    file_path: Path


class ExecutionSpecLoader:
    """Load execution layer specs from ethereum/execution-specs repository."""

    REPO_URL = "https://github.com/ethereum/execution-specs.git"
    FORKS = [
        "frontier",
        "homestead",
        "tangerine_whistle",
        "spurious_dragon",
        "byzantium",
        "constantinople",
        "istanbul",
        "berlin",
        "london",
        "arrow_glacier",
        "gray_glacier",
        "paris",
        "shanghai",
        "cancun",
        "prague",
    ]

    def __init__(self, repo_path: str | Path = "data/execution-specs") -> None:
        self.repo_path = Path(repo_path)

    def clone_or_update(self) -> str:
        """Clone or update the execution-specs repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_execution_specs", url=self.REPO_URL)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "100", self.REPO_URL, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_execution_specs")
            subprocess.run(
                ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_all_specs(self) -> list[ExecutionSpec]:
        """Load all Python specification files from the repository."""
        specs: list[ExecutionSpec] = []

        src_dir = self.repo_path / "src" / "ethereum"
        if not src_dir.exists():
            logger.warning("src_dir_not_found", path=str(src_dir))
            return specs

        for fork in self.FORKS:
            fork_dir = src_dir / fork
            if not fork_dir.exists():
                continue

            for py_file in fork_dir.rglob("*.py"):
                # Skip __pycache__ and test files
                if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
                    continue

                spec = self._parse_spec_file(py_file, fork)
                if spec:
                    specs.append(spec)

        logger.info("loaded_execution_specs", count=len(specs))
        return specs

    def _parse_spec_file(self, file_path: Path, fork: str) -> ExecutionSpec | None:
        """Parse a single Python specification file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_spec", path=str(file_path), error=str(e))
            return None

        # Skip empty or very small files
        if len(content) < 100:
            return None

        # Build module path from file path
        rel_path = file_path.relative_to(self.repo_path / "src")
        module_path = str(rel_path).replace("/", ".").replace(".py", "")

        return ExecutionSpec(
            module_path=module_path,
            fork=fork,
            content=content,
            file_path=file_path,
        )
```

### 4.2 Export the loader

Edit `src/ingestion/__init__.py` to add:

```python
from src.ingestion.execution_spec_loader import ExecutionSpec, ExecutionSpecLoader
```

And add to `__all__`.

**Phase 4 validation:**
```bash
uv run python -c "
from src.ingestion import ExecutionSpecLoader

loader = ExecutionSpecLoader()
commit = loader.clone_or_update()
print(f'Commit: {commit}')
specs = loader.load_all_specs()
print(f'Loaded {len(specs)} execution specs')
for spec in specs[:3]:
    print(f'  - {spec.fork}: {spec.module_path}')
"
```

---

## Phase 5: Create adaptive rate limiter

### 5.1 Create rate limiter module

Create file: `src/ingestion/rate_limiter.py`

```python
"""Adaptive rate limiter with header parsing and exponential backoff."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitState:
    """Tracks rate limit state for a specific API."""

    requests_made: int = 0
    window_start: float = field(default_factory=time.monotonic)
    window_seconds: float = 60.0

    # Discovered limits (learned from headers or 429 responses)
    discovered_limit: int | None = None
    discovered_remaining: int | None = None
    discovered_reset_time: float | None = None

    # Backoff state
    consecutive_429s: int = 0
    last_429_time: float | None = None
    current_backoff_seconds: float = 1.0

    # Default conservative limits (used when no headers available)
    default_requests_per_minute: int = 30

    def reset_window_if_expired(self) -> None:
        """Reset the request window if it has expired."""
        now = time.monotonic()
        if now - self.window_start >= self.window_seconds:
            self.requests_made = 0
            self.window_start = now

    def record_request(self) -> None:
        """Record that a request was made."""
        self.reset_window_if_expired()
        self.requests_made += 1

    def should_wait(self) -> tuple[bool, float]:
        """Check if we should wait before making a request.

        Returns:
            Tuple of (should_wait, wait_seconds)
        """
        now = time.monotonic()

        # Check if we're in a backoff period from a 429
        if self.last_429_time and self.current_backoff_seconds > 0:
            elapsed = now - self.last_429_time
            if elapsed < self.current_backoff_seconds:
                return True, self.current_backoff_seconds - elapsed

        # Check discovered reset time
        if self.discovered_reset_time and now < self.discovered_reset_time:
            if self.discovered_remaining is not None and self.discovered_remaining <= 0:
                return True, self.discovered_reset_time - now

        # Check our own accounting against discovered or default limits
        self.reset_window_if_expired()
        limit = self.discovered_limit or self.default_requests_per_minute

        if self.requests_made >= limit:
            wait_time = self.window_seconds - (now - self.window_start)
            if wait_time > 0:
                return True, wait_time

        return False, 0.0

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update state from rate limit response headers.

        Handles common header formats:
        - X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
        - RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset
        - Retry-After
        """
        # Try different header naming conventions
        limit_keys = ["X-RateLimit-Limit", "RateLimit-Limit", "x-ratelimit-limit"]
        remaining_keys = ["X-RateLimit-Remaining", "RateLimit-Remaining", "x-ratelimit-remaining"]
        reset_keys = ["X-RateLimit-Reset", "RateLimit-Reset", "x-ratelimit-reset"]

        # Case-insensitive header lookup
        lower_headers = {k.lower(): v for k, v in headers.items()}

        for key in limit_keys:
            if key.lower() in lower_headers:
                try:
                    self.discovered_limit = int(lower_headers[key.lower()])
                    logger.debug("discovered_rate_limit", limit=self.discovered_limit)
                except ValueError:
                    pass
                break

        for key in remaining_keys:
            if key.lower() in lower_headers:
                try:
                    self.discovered_remaining = int(lower_headers[key.lower()])
                except ValueError:
                    pass
                break

        for key in reset_keys:
            if key.lower() in lower_headers:
                try:
                    reset_value = lower_headers[key.lower()]
                    # Could be Unix timestamp or seconds until reset
                    reset_int = int(reset_value)
                    if reset_int > 1000000000:  # Looks like Unix timestamp
                        self.discovered_reset_time = float(reset_int)
                    else:  # Seconds until reset
                        self.discovered_reset_time = time.time() + reset_int
                except ValueError:
                    pass
                break

    def handle_429(self, headers: dict[str, str] | None = None) -> float:
        """Handle a 429 response and return wait time.

        Args:
            headers: Response headers (may contain Retry-After)

        Returns:
            Seconds to wait before retrying
        """
        self.consecutive_429s += 1
        self.last_429_time = time.monotonic()

        # Check for Retry-After header
        if headers:
            lower_headers = {k.lower(): v for k, v in headers.items()}
            retry_after = lower_headers.get("retry-after")
            if retry_after:
                try:
                    wait_time = float(retry_after)
                    self.current_backoff_seconds = wait_time
                    logger.info("rate_limited_with_retry_after", wait_seconds=wait_time)
                    return wait_time
                except ValueError:
                    pass

        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s max
        self.current_backoff_seconds = min(60.0, 2 ** (self.consecutive_429s - 1))
        logger.warning(
            "rate_limited_exponential_backoff",
            consecutive_429s=self.consecutive_429s,
            backoff_seconds=self.current_backoff_seconds,
        )
        return self.current_backoff_seconds

    def handle_success(self) -> None:
        """Handle a successful response - reset backoff state."""
        if self.consecutive_429s > 0:
            logger.debug("rate_limit_backoff_cleared", previous_429s=self.consecutive_429s)
        self.consecutive_429s = 0
        self.current_backoff_seconds = 1.0


class AdaptiveRateLimiter:
    """Manages rate limiting across multiple API endpoints."""

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()

    def _get_state(self, api_name: str, default_rpm: int = 30) -> RateLimitState:
        """Get or create rate limit state for an API."""
        if api_name not in self._states:
            self._states[api_name] = RateLimitState(default_requests_per_minute=default_rpm)
        return self._states[api_name]

    async def wait_if_needed(self, api_name: str, default_rpm: int = 30) -> None:
        """Wait if rate limit requires it.

        Args:
            api_name: Identifier for the API (e.g., "ethresearch", "arxiv")
            default_rpm: Default requests per minute if no limit discovered
        """
        async with self._lock:
            state = self._get_state(api_name, default_rpm)
            should_wait, wait_time = state.should_wait()

        if should_wait and wait_time > 0:
            logger.info("rate_limit_waiting", api=api_name, seconds=round(wait_time, 1))
            await asyncio.sleep(wait_time)

    async def record_request(self, api_name: str) -> None:
        """Record that a request was made."""
        async with self._lock:
            state = self._get_state(api_name)
            state.record_request()

    async def update_from_response(
        self,
        api_name: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> float:
        """Update rate limit state from response.

        Args:
            api_name: API identifier
            status_code: HTTP response status code
            headers: Response headers

        Returns:
            Wait time in seconds (0 if no wait needed, >0 if rate limited)
        """
        async with self._lock:
            state = self._get_state(api_name)

            if headers:
                state.update_from_headers(headers)

            if status_code == 429:
                return state.handle_429(headers)
            else:
                state.handle_success()
                return 0.0

    async def execute_with_retry(
        self,
        api_name: str,
        func: Any,
        *args: Any,
        max_retries: int = 5,
        default_rpm: int = 30,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with automatic rate limit handling.

        Args:
            api_name: API identifier for rate limiting
            func: Async function to execute (should return (status_code, headers, result))
            max_retries: Maximum retry attempts on 429
            default_rpm: Default requests per minute

        Returns:
            Result from the function

        Raises:
            Exception: If max retries exceeded or non-429 error
        """
        for attempt in range(max_retries + 1):
            await self.wait_if_needed(api_name, default_rpm)
            await self.record_request(api_name)

            try:
                status_code, headers, result = await func(*args, **kwargs)

                wait_time = await self.update_from_response(api_name, status_code, headers)

                if status_code == 429:
                    if attempt < max_retries:
                        logger.warning(
                            "rate_limit_retry",
                            api=api_name,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_seconds=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")

                return result

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = await self.update_from_response(api_name, 429, None)
                    if attempt < max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                raise

        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")


# Global rate limiter instance for shared use
_global_limiter: AdaptiveRateLimiter | None = None


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = AdaptiveRateLimiter()
    return _global_limiter
```

### 5.2 Export the rate limiter

Edit `src/ingestion/__init__.py` to add:

```python
from src.ingestion.rate_limiter import (
    AdaptiveRateLimiter,
    RateLimitState,
    get_rate_limiter,
)
```

And add to `__all__`:
```python
"AdaptiveRateLimiter",
"RateLimitState",
"get_rate_limiter",
```

### 5.3 Update existing loaders to use rate limiter

#### Update `src/ingestion/discourse_client.py`

Find the `_request` method and wrap it with rate limiting:

```python
from src.ingestion.rate_limiter import get_rate_limiter

class DiscourseClient:
    async def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a rate-limited request to the Discourse API."""
        limiter = get_rate_limiter()

        async def do_request() -> tuple[int, dict[str, str], dict]:
            async with self._session.get(
                f"{self.base_url}/{endpoint}.json",
                params=params,
            ) as response:
                headers = dict(response.headers)
                if response.status == 429:
                    return 429, headers, {}
                response.raise_for_status()
                return response.status, headers, await response.json()

        return await limiter.execute_with_retry(
            api_name=self._api_name,  # "ethresearch" or "magicians"
            func=do_request,
            default_rpm=20,  # Discourse default is ~20 req/min
        )
```

#### Update `src/ingestion/arxiv_fetcher.py`

```python
from src.ingestion.rate_limiter import get_rate_limiter

class ArxivFetcher:
    def __init__(self) -> None:
        self._limiter = get_rate_limiter()

    async def _fetch_page(self, query: str, start: int, max_results: int) -> list[ArxivPaper]:
        """Fetch a page of results with rate limiting."""

        async def do_fetch() -> tuple[int, dict[str, str], list[ArxivPaper]]:
            # arXiv API call here
            response = await self._session.get(...)
            headers = dict(response.headers)

            if response.status == 429:
                return 429, headers, []

            # Parse results...
            return response.status, headers, papers

        return await self._limiter.execute_with_retry(
            api_name="arxiv",
            func=do_fetch,
            default_rpm=10,  # arXiv recommends max 1 req/3sec = 20/min, be conservative
        )
```

**Phase 5 validation:**
```bash
uv run python -c "
import asyncio
from src.ingestion.rate_limiter import AdaptiveRateLimiter, RateLimitState

async def test():
    # Test RateLimitState
    state = RateLimitState(default_requests_per_minute=5)

    # Simulate requests
    for i in range(6):
        state.record_request()
        should_wait, wait_time = state.should_wait()
        print(f'Request {i+1}: should_wait={should_wait}, wait_time={wait_time:.1f}')

    # Test header parsing
    state.update_from_headers({
        'X-RateLimit-Limit': '100',
        'X-RateLimit-Remaining': '50',
    })
    print(f'Discovered limit: {state.discovered_limit}')
    print(f'Discovered remaining: {state.discovered_remaining}')

    # Test 429 handling
    wait = state.handle_429({'Retry-After': '30'})
    print(f'429 with Retry-After: wait={wait}s')

    # Test limiter
    limiter = AdaptiveRateLimiter()
    await limiter.wait_if_needed('test-api', default_rpm=60)
    await limiter.record_request('test-api')
    wait = await limiter.update_from_response('test-api', 200, {'X-RateLimit-Remaining': '99'})
    print(f'After success: wait={wait}')

    print('Rate limiter tests passed!')

asyncio.run(test())
"
```

---

## Phase 6: Create chunk converter utility

### 6.1 Create converter file

Create file: `src/chunking/chunk_converter.py`

```python
"""Utility to convert specialized chunk types to standard Chunk format."""

from src.chunking.chunker import Chunk


def to_standard_chunk(
    chunk: object,
    document_id: str,
) -> Chunk:
    """Convert a specialized chunk to standard Chunk format.

    Handles: PaperChunk, TranscriptChunk, CodeChunk, and any chunk with
    chunk_id, content, token_count, chunk_index attributes.
    """
    # Get section path from various possible attributes
    section_path = None
    for attr in ["section", "speaker", "function_name", "section_path"]:
        if hasattr(chunk, attr):
            section_path = getattr(chunk, attr)
            if section_path:
                break

    return Chunk(
        chunk_id=getattr(chunk, "chunk_id", f"{document_id}-{getattr(chunk, 'chunk_index', 0)}"),
        document_id=document_id,
        content=chunk.content,
        token_count=getattr(chunk, "token_count", len(chunk.content.split())),
        chunk_index=getattr(chunk, "chunk_index", 0),
        section_path=section_path,
    )


def convert_chunks(
    chunks: list,
    document_id: str,
) -> list[Chunk]:
    """Convert a list of specialized chunks to standard Chunks."""
    return [to_standard_chunk(c, document_id) for c in chunks]
```

### 6.2 Export the converter

Edit `src/chunking/__init__.py` to add:

```python
from src.chunking.chunk_converter import to_standard_chunk, convert_chunks
```

**Phase 5 validation:**
```bash
uv run python -c "
from src.chunking import convert_chunks, Chunk
from dataclasses import dataclass

@dataclass
class TestChunk:
    chunk_id: str
    content: str
    token_count: int
    chunk_index: int
    section: str

test = TestChunk('test-1', 'Hello world', 2, 0, 'intro')
result = convert_chunks([test], 'doc-1')
print(f'Converted: {result[0]}')
assert isinstance(result[0], Chunk)
print('Chunk converter working!')
"
```

---

## Phase 7: Create ingestion scripts

**IMPORTANT:** All ingestion scripts MUST use the `AdaptiveRateLimiter` from Phase 5 for API calls.

### 7.1 Create ethresear.ch ingestion script

Create file: `scripts/ingest_ethresearch.py`

```python
#!/usr/bin/env python3
"""Ingest ethresear.ch forum topics into the database.

Usage:
    uv run python scripts/ingest_ethresearch.py [--max-topics 1000] [--max-pages 100]
"""

import argparse
import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import ForumChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import EthresearchLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


async def ingest_ethresearch(max_topics: int = 1000, max_pages: int = 100) -> None:
    """Ingest ethresear.ch topics into the database."""
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()

    chunker = ForumChunker(max_tokens=512)
    topics_ingested = 0
    chunks_stored = 0

    try:
        async with EthresearchLoader() as loader:
            async for topic in loader.iter_topics_with_posts(max_topics=max_topics):
                document_id = f"ethresearch-topic-{topic.topic_id}"

                # Check if already ingested
                # (implement check or use upsert)

                # Chunk the topic
                topic_chunks = chunker.chunk_topic(topic)
                if not topic_chunks:
                    continue

                # Convert to standard chunks
                standard_chunks = convert_chunks(topic_chunks, document_id)

                # Embed chunks
                embedded = embedder.embed_chunks(standard_chunks)

                # Store document metadata
                await store.store_generic_document(
                    document_id=document_id,
                    document_type="forum_topic",
                    title=topic.title,
                    source="ethresearch",
                    raw_content="\n\n".join(p.content for p in topic.posts),
                    metadata={
                        "topic_id": topic.topic_id,
                        "category": topic.category,
                        "tags": getattr(topic, "tags", []),
                        "posts_count": len(topic.posts),
                    },
                )

                # Store chunks
                await store.store_embedded_chunks(embedded, document_id=document_id)

                topics_ingested += 1
                chunks_stored += len(embedded)
                logger.info(
                    "ingested_topic",
                    topic_id=topic.topic_id,
                    title=topic.title[:50],
                    chunks=len(embedded),
                )

    finally:
        await store.close()

    logger.info(
        "ethresearch_ingestion_complete",
        topics=topics_ingested,
        chunks=chunks_stored,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ethresear.ch topics")
    parser.add_argument("--max-topics", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(ingest_ethresearch(max_topics=args.max_topics, max_pages=args.max_pages))


if __name__ == "__main__":
    main()
```

### 7.2 Create Ethereum Magicians ingestion script

Create file: `scripts/ingest_magicians.py`

Similar structure to ethresearch, using `MagiciansLoader`.

### 7.3 Create ACD transcripts ingestion script

Create file: `scripts/ingest_acd_transcripts.py`

```python
#!/usr/bin/env python3
"""Ingest AllCoreDevs call transcripts.

Usage:
    uv run python scripts/ingest_acd_transcripts.py
"""

import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import TranscriptChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import ACDTranscriptLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


async def ingest_acd_transcripts() -> None:
    """Ingest ACD transcripts into the database."""
    loader = ACDTranscriptLoader()
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()

    chunker = TranscriptChunker(max_tokens=512)

    try:
        # Clone/update ethereum/pm repo
        loader.clone_repo()

        transcripts = loader.list_transcripts()
        logger.info("found_transcripts", count=len(transcripts))

        for transcript in transcripts:
            document_id = f"acd-call-{transcript.call_number}"

            # Chunk the transcript
            transcript_chunks = chunker.chunk(transcript)
            if not transcript_chunks:
                continue

            # Convert to standard chunks
            standard_chunks = convert_chunks(transcript_chunks, document_id)

            # Embed chunks
            embedded = embedder.embed_chunks(standard_chunks)

            # Store document
            await store.store_generic_document(
                document_id=document_id,
                document_type="acd_transcript",
                title=transcript.title or f"ACD Call #{transcript.call_number}",
                source="ethereum/pm",
                raw_content=transcript.raw_markdown,
                metadata={
                    "call_number": transcript.call_number,
                    "date": transcript.date.isoformat() if transcript.date else None,
                    "speakers": transcript.speakers,
                },
            )

            # Store chunks
            await store.store_embedded_chunks(embedded, document_id=document_id)

            logger.info(
                "ingested_transcript",
                call_number=transcript.call_number,
                chunks=len(embedded),
            )

    finally:
        await store.close()

    logger.info("acd_ingestion_complete")


def main() -> None:
    asyncio.run(ingest_acd_transcripts())


if __name__ == "__main__":
    main()
```

### 7.4 Create arXiv ingestion script

Create file: `scripts/ingest_arxiv.py`

```python
#!/usr/bin/env python3
"""Ingest Ethereum-related arXiv papers.

Usage:
    uv run python scripts/ingest_arxiv.py [--max-papers 300]
"""

import argparse
import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import PaperChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import ArxivFetcher, PDFExtractor, QualityScorer
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


async def ingest_arxiv(max_papers: int = 300) -> None:
    """Ingest arXiv papers into the database."""
    fetcher = ArxivFetcher()
    extractor = PDFExtractor()
    scorer = QualityScorer()
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()

    chunker = PaperChunker(max_tokens=512)
    papers_ingested = 0
    papers_skipped = 0

    try:
        papers = fetcher.search_ethereum_papers(max_results=max_papers)
        logger.info("found_papers", count=len(papers))

        for paper in papers:
            if not paper.pdf_url:
                papers_skipped += 1
                continue

            document_id = f"arxiv-{paper.arxiv_id}"

            try:
                # Extract PDF content
                pdf_content = extractor.extract_from_url(paper.pdf_url)

                # Check quality
                quality = scorer.score(pdf_content)
                if not scorer.is_acceptable(quality, threshold=0.5):
                    logger.warning(
                        "low_quality_pdf",
                        arxiv_id=paper.arxiv_id,
                        score=quality.overall,
                    )
                    papers_skipped += 1
                    continue

            except Exception as e:
                logger.warning("pdf_extraction_failed", arxiv_id=paper.arxiv_id, error=str(e))
                papers_skipped += 1
                continue

            # Chunk the paper
            paper_chunks = chunker.chunk(pdf_content)
            if not paper_chunks:
                papers_skipped += 1
                continue

            # Convert to standard chunks
            standard_chunks = convert_chunks(paper_chunks, document_id)

            # Embed chunks
            embedded = embedder.embed_chunks(standard_chunks)

            # Store document
            await store.store_generic_document(
                document_id=document_id,
                document_type="arxiv_paper",
                title=paper.title,
                source="arxiv",
                author=", ".join(paper.authors),
                raw_content=pdf_content.full_text,
                metadata={
                    "arxiv_id": paper.arxiv_id,
                    "categories": paper.categories,
                    "published": paper.published.isoformat() if paper.published else None,
                    "doi": paper.doi,
                    "abstract": paper.abstract,
                },
            )

            # Store chunks
            await store.store_embedded_chunks(embedded, document_id=document_id)

            papers_ingested += 1
            logger.info(
                "ingested_paper",
                arxiv_id=paper.arxiv_id,
                title=paper.title[:50],
                chunks=len(embedded),
            )

    finally:
        await store.close()

    logger.info(
        "arxiv_ingestion_complete",
        ingested=papers_ingested,
        skipped=papers_skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers")
    parser.add_argument("--max-papers", type=int, default=300)
    args = parser.parse_args()

    asyncio.run(ingest_arxiv(max_papers=args.max_papers))


if __name__ == "__main__":
    main()
```

### 7.5 Create consensus specs ingestion script

Create file: `scripts/ingest_consensus_specs.py`

```python
#!/usr/bin/env python3
"""Ingest ethereum/consensus-specs repository.

Usage:
    uv run python scripts/ingest_consensus_specs.py
"""

import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import SectionChunker
from src.embeddings import VoyageEmbedder
from src.ingestion import ConsensusSpecLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


async def ingest_consensus_specs() -> None:
    """Ingest consensus specs into the database."""
    loader = ConsensusSpecLoader()
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()

    chunker = SectionChunker(max_tokens=512)

    try:
        git_commit = loader.clone_or_update()
        specs = loader.load_all_specs()
        logger.info("found_specs", count=len(specs))

        for spec in specs:
            document_id = f"consensus-spec-{spec.fork}-{spec.name}"

            # Chunk using section chunker with markdown content
            chunks = chunker.chunk_markdown(spec.content, document_id)
            if not chunks:
                continue

            # Embed chunks
            embedded = embedder.embed_chunks(chunks)

            # Store document
            await store.store_generic_document(
                document_id=document_id,
                document_type="consensus_spec",
                title=spec.title,
                source="ethereum/consensus-specs",
                raw_content=spec.content,
                git_commit=git_commit,
                metadata={
                    "fork": spec.fork,
                    "spec_name": spec.name,
                    "file_path": str(spec.file_path),
                },
            )

            # Store chunks
            await store.store_embedded_chunks(embedded, document_id=document_id, git_commit=git_commit)

            logger.info(
                "ingested_spec",
                fork=spec.fork,
                name=spec.name,
                chunks=len(embedded),
            )

    finally:
        await store.close()

    logger.info("consensus_specs_ingestion_complete")


def main() -> None:
    asyncio.run(ingest_consensus_specs())


if __name__ == "__main__":
    main()
```

### 7.6 Create execution specs ingestion script

Create file: `scripts/ingest_execution_specs.py`

Similar to consensus specs, using `ExecutionSpecLoader` and `CodeChunker`.

### 7.7 Create master ingestion script

Create file: `scripts/ingest_all.py`

```python
#!/usr/bin/env python3
"""Ingest all data sources into the database.

Usage:
    uv run python scripts/ingest_all.py [--skip-eips] [--skip-forums] [--skip-transcripts] [--skip-papers] [--skip-specs]
"""

import argparse
import asyncio
import subprocess
import sys

import structlog

logger = structlog.get_logger()


async def run_script(script_path: str, args: list[str] | None = None) -> bool:
    """Run an ingestion script and return success status."""
    cmd = ["uv", "run", "python", script_path]
    if args:
        cmd.extend(args)

    logger.info("running_script", script=script_path)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


async def ingest_all(
    skip_eips: bool = False,
    skip_forums: bool = False,
    skip_transcripts: bool = False,
    skip_papers: bool = False,
    skip_specs: bool = False,
) -> None:
    """Run all ingestion scripts."""
    results = {}

    if not skip_eips:
        logger.info("phase", name="EIPs")
        results["eips"] = await run_script("scripts/ingest_eips.py")

    if not skip_forums:
        logger.info("phase", name="ethresear.ch")
        results["ethresearch"] = await run_script("scripts/ingest_ethresearch.py", ["--max-topics", "1000"])

        logger.info("phase", name="Ethereum Magicians")
        results["magicians"] = await run_script("scripts/ingest_magicians.py", ["--max-topics", "1000"])

    if not skip_transcripts:
        logger.info("phase", name="ACD Transcripts")
        results["transcripts"] = await run_script("scripts/ingest_acd_transcripts.py")

    if not skip_papers:
        logger.info("phase", name="arXiv Papers")
        results["arxiv"] = await run_script("scripts/ingest_arxiv.py", ["--max-papers", "300"])

    if not skip_specs:
        logger.info("phase", name="Consensus Specs")
        results["consensus"] = await run_script("scripts/ingest_consensus_specs.py")

        logger.info("phase", name="Execution Specs")
        results["execution"] = await run_script("scripts/ingest_execution_specs.py")

    # Summary
    logger.info("ingestion_complete", results=results)

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.error("some_ingestions_failed", failed=failed)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all data sources")
    parser.add_argument("--skip-eips", action="store_true")
    parser.add_argument("--skip-forums", action="store_true")
    parser.add_argument("--skip-transcripts", action="store_true")
    parser.add_argument("--skip-papers", action="store_true")
    parser.add_argument("--skip-specs", action="store_true")
    args = parser.parse_args()

    asyncio.run(ingest_all(
        skip_eips=args.skip_eips,
        skip_forums=args.skip_forums,
        skip_transcripts=args.skip_transcripts,
        skip_papers=args.skip_papers,
        skip_specs=args.skip_specs,
    ))


if __name__ == "__main__":
    main()
```

**Phase 7 validation:**
```bash
# Verify scripts are syntactically correct
uv run python -m py_compile scripts/ingest_ethresearch.py
uv run python -m py_compile scripts/ingest_magicians.py
uv run python -m py_compile scripts/ingest_acd_transcripts.py
uv run python -m py_compile scripts/ingest_arxiv.py
uv run python -m py_compile scripts/ingest_consensus_specs.py
uv run python -m py_compile scripts/ingest_execution_specs.py
uv run python -m py_compile scripts/ingest_all.py
```

---

## Phase 8: Run full ingestion

### 8.1 Ingest EIPs first

```bash
uv run python scripts/ingest_eips.py
```

Wait for completion. Expected: ~900 EIPs, ~10,000 chunks, ~2 minutes.

### 8.2 Ingest remaining sources

```bash
uv run python scripts/ingest_all.py --skip-eips
```

This will take ~2-3 hours due to rate limiting. Monitor progress via logs.

**Alternatively, run individually for better control:**

```bash
# Forums (longest, rate limited)
uv run python scripts/ingest_ethresearch.py --max-topics 500
uv run python scripts/ingest_magicians.py --max-topics 500

# Transcripts (fast)
uv run python scripts/ingest_acd_transcripts.py

# Papers (moderate, PDF download)
uv run python scripts/ingest_arxiv.py --max-papers 200

# Specs (fast)
uv run python scripts/ingest_consensus_specs.py
uv run python scripts/ingest_execution_specs.py
```

---

## Phase 9: Create validation script

Create file: `scripts/validate_corpus.py`

```python
#!/usr/bin/env python3
"""Validate full corpus and test retrieval quality.

Usage:
    uv run python scripts/validate_corpus.py
"""

import asyncio

import structlog
from dotenv import load_dotenv

from src.embeddings import VoyageEmbedder
from src.generation import CitedGenerator
from src.retrieval import SimpleRetriever
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


TEST_QUERIES = [
    "What are the different approaches that have been suggested over time for encrypted mempools?",
    "What are the upstream dependencies of ZK EVMs?",
    "Why was RISC-V chosen over Wasm?",
]


async def validate_corpus() -> None:
    """Validate corpus ingestion and query capabilities."""
    store = PgVectorStore()
    await store.connect()

    embedder = VoyageEmbedder()
    retriever = SimpleRetriever(embedder, store)

    print("\n" + "=" * 60)
    print("CORPUS VALIDATION")
    print("=" * 60)

    # 1. Get document stats
    print("\n1. Document counts by type:")
    async with store._pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT document_type, COUNT(*) as count
            FROM documents
            GROUP BY document_type
            ORDER BY count DESC
        """)
        for row in rows:
            print(f"   {row['document_type']}: {row['count']}")

    # 2. Get chunk counts
    print("\n2. Chunk counts:")
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as count FROM chunks")
        print(f"   Total chunks: {row['count']}")

    # 3. Test retrieval for each query
    print("\n3. Retrieval test results:")
    for query in TEST_QUERIES:
        print(f"\n   Query: {query[:60]}...")
        results = await retriever.retrieve(query, top_k=5)

        sources = set()
        for r in results.results:
            doc_id = r.chunk.document_id
            source_type = doc_id.split("-")[0] if "-" in doc_id else "eip"
            sources.add(source_type)

        print(f"   Sources found: {sources}")
        print(f"   Top result: {results.results[0].chunk.document_id if results.results else 'None'}")

    # 4. Test generation (if API key available)
    print("\n4. Generation test:")
    try:
        import anthropic
        client = anthropic.Anthropic()
        generator = CitedGenerator(client, retriever)

        test_q = TEST_QUERIES[0]
        result = await generator.generate(test_q)
        print(f"   Query: {test_q[:50]}...")
        print(f"   Response length: {len(result.response)} chars")
        print(f"   Sources cited: {len(result.retrieval.results)}")
        print(f"   First 200 chars: {result.response[:200]}...")
    except Exception as e:
        print(f"   Generation test skipped: {e}")

    await store.close()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60 + "\n")


def main() -> None:
    asyncio.run(validate_corpus())


if __name__ == "__main__":
    main()
```

**Phase 9 validation:**
```bash
uv run python scripts/validate_corpus.py
```

---

## Phase 10: Final verification

Run the test queries manually:

```bash
# Query 1: Encrypted mempools
uv run python scripts/query_cli.py "What are the different approaches that have been suggested over time for encrypted mempools?" --mode cited --show-sources

# Query 2: ZK EVM dependencies
uv run python scripts/query_cli.py "What are the upstream dependencies of ZK EVMs?" --mode cited --show-sources

# Query 3: RISC-V vs Wasm
uv run python scripts/query_cli.py "Why was RISC-V chosen over Wasm?" --mode cited --show-sources
```

**Success criteria:**
- Each query returns a coherent, factual answer
- Sources include multiple document types
- No hallucinated information (all claims backed by retrieved context)

---

## Troubleshooting guide

### Database connection fails

```bash
docker compose down && docker compose up -d
sleep 10
docker compose ps
```

### Import errors

```bash
uv sync
uv run ruff check . --fix
```

### Rate limiting errors (forums and APIs)

The `AdaptiveRateLimiter` handles most rate limiting automatically, but you may need to adjust settings:

**Symptoms:**
- Repeated 429 errors in logs
- "Rate limit exceeded after N retries" errors
- Long pauses between requests

**Diagnosis:**
```bash
# Check rate limiter state in logs
uv run python scripts/ingest_ethresearch.py --max-topics 10 2>&1 | grep -i rate
```

**Solutions:**

1. **Reduce concurrency** - Process fewer topics at once:
```bash
uv run python scripts/ingest_ethresearch.py --max-topics 100
```

2. **Increase default backoff** - If APIs are stricter than expected, modify the rate limiter defaults in `src/ingestion/rate_limiter.py`:
```python
# More conservative defaults
default_requests_per_minute: int = 15  # Was 30
```

3. **Check for Retry-After headers** - The limiter respects these automatically. If you see consistent wait times, the API is telling us exactly how long to wait.

4. **API-specific limits:**
   - **Discourse (ethresear.ch, Magicians)**: ~20 req/min default, may have stricter limits for unauthenticated users
   - **arXiv**: Recommends max 1 request per 3 seconds (~20/min), but be conservative with 10/min
   - **Voyage AI**: Check your plan limits at https://dash.voyageai.com

5. **Manual retry after cooldown:**
```bash
# Wait 5 minutes, then resume with smaller batch
sleep 300 && uv run python scripts/ingest_ethresearch.py --max-topics 50
```

### PDF extraction fails

Skip problematic papers (already handled with try/except). Check logs for skipped papers.

### Out of memory

Reduce batch sizes in all ingestion scripts. Process fewer documents at a time.

### Voyage API errors

Check API key:
```bash
echo $VOYAGE_API_KEY
```

Check rate limits - add delays between batches if needed.

---

## Files summary

### New files to create

1. `src/ingestion/consensus_spec_loader.py`
2. `src/ingestion/execution_spec_loader.py`
3. `src/ingestion/rate_limiter.py` (adaptive rate limiting with header parsing)
4. `src/chunking/chunk_converter.py`
5. `scripts/ingest_ethresearch.py`
6. `scripts/ingest_magicians.py`
7. `scripts/ingest_acd_transcripts.py`
8. `scripts/ingest_arxiv.py`
9. `scripts/ingest_consensus_specs.py`
10. `scripts/ingest_execution_specs.py`
11. `scripts/ingest_all.py`
12. `scripts/validate_corpus.py`

### Files to modify

1. `src/storage/pg_vector_store.py` - Add `store_generic_document()`, update schema
2. `src/ingestion/__init__.py` - Export new loaders and rate limiter
3. `src/ingestion/discourse_client.py` - Integrate rate limiter
4. `src/ingestion/arxiv_fetcher.py` - Integrate rate limiter
5. `src/chunking/__init__.py` - Export chunk converter

---

## Estimated timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Prerequisites verification | 5 min |
| 2 | Database schema enhancement | 30 min |
| 3 | Consensus spec loader | 30 min |
| 4 | Execution spec loader | 30 min |
| 5 | Adaptive rate limiter | 45 min |
| 6 | Chunk converter | 15 min |
| 7 | Ingestion scripts | 2 hours |
| 8 | Run full ingestion | 3 hours |
| 9 | Validation script | 30 min |
| 10 | Final verification | 30 min |
| **Total** | | **~9 hours** |

Most time is spent waiting for ingestion to complete (rate-limited API calls).

---

## Rate limiting architecture

### Design principles

1. **Header-first**: Always check response headers for rate limit info
2. **Learn and adapt**: Discovered limits override conservative defaults
3. **Exponential backoff**: On 429, back off 1s → 2s → 4s → 8s → ... → 60s max
4. **Respect Retry-After**: If server tells us when to retry, obey exactly
5. **Per-API tracking**: Each API (ethresearch, arxiv, voyage) has independent state

### Header formats supported

```
X-RateLimit-Limit: 100        # Max requests per window
X-RateLimit-Remaining: 50     # Requests left in window
X-RateLimit-Reset: 1699999999 # Unix timestamp when window resets
Retry-After: 30               # Seconds to wait (on 429)
```

### Flow diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Request Flow                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │ Request │───►│ wait_if_     │───►│ record_request  │     │
│  │         │    │ needed()     │    │                 │     │
│  └─────────┘    └──────────────┘    └────────┬────────┘     │
│                        │                      │              │
│                   Wait if:                    ▼              │
│                   - In backoff           ┌─────────┐        │
│                   - At limit             │ Execute │        │
│                   - Reset pending        │   API   │        │
│                                          └────┬────┘        │
│                                               │              │
│                                               ▼              │
│                                    ┌──────────────────┐     │
│                                    │ update_from_     │     │
│                                    │ response()       │     │
│                                    └────────┬─────────┘     │
│                                             │               │
│                          ┌──────────────────┼───────────┐   │
│                          │                  │           │   │
│                          ▼                  ▼           ▼   │
│                    ┌──────────┐      ┌──────────┐  ┌──────┐│
│                    │ 200 OK   │      │ 429      │  │ Error││
│                    │          │      │          │  │      ││
│                    │ Parse    │      │ Backoff  │  │Raise ││
│                    │ headers  │      │ + Retry  │  │      ││
│                    │ Clear    │      │          │  │      ││
│                    │ backoff  │      │          │  │      ││
│                    └──────────┘      └──────────┘  └──────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Default rate limits (conservative)

| API | Default RPM | Notes |
|-----|-------------|-------|
| ethresear.ch | 20 | Discourse forum |
| Ethereum Magicians | 20 | Discourse forum |
| arXiv | 10 | Recommends 1 req/3s, be conservative |
| Voyage AI | 60 | Check your plan |
| GitHub (specs) | 60 | Unauthenticated limit |
