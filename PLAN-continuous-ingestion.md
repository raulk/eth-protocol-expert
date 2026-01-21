# Continuous ingestion plan

## Overview

Keep the Ethereum Protocol Expert corpus up-to-date with new content from all sources. Different sources have different update patterns and require tailored sync strategies.

## Source characteristics

| Source | Update Frequency | API Support | Change Detection | Priority |
|--------|------------------|-------------|------------------|----------|
| GitHub repos | Daily (specs), hourly (issues) | REST + GraphQL | Webhooks, `since` param | High |
| arXiv papers | Weekly | REST | Search by date | Medium |
| Ethresearch | Daily | Discourse API | `latest.json` | High |
| Magicians | Daily | Discourse API | `latest.json` | Medium |
| ACD transcripts | Weekly | GitHub (pm repo) | Git commits | High |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ingestion Orchestrator                           │
│  - Schedules sync jobs per source                                       │
│  - Tracks sync state (last_sync, last_id, cursors)                      │
│  - Handles failures and retries                                         │
│  - Emits metrics/alerts                                                 │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  GitHub Syncer  │ │  arXiv Syncer   │ │ Discourse Syncer│ │  Custom Syncer  │
│  - Files        │ │  - Search API   │ │ - Ethresearch   │ │  - Future       │
│  - Issues/PRs   │ │  - PDF fetch    │ │ - Magicians     │ │    sources      │
│  - Webhooks     │ │  - Weekly batch │ │ - Rate limited  │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Sync State Store                               │
│  data/sync_state.json                                                   │
│  - Per-source cursors (last_sync, last_id, etag)                        │
│  - Failure counts and backoff state                                     │
│  - Metrics (docs synced, errors, duration)                              │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Processing Pipeline                              │
│  Cache → Domain Processing → Chunking → Embedding → Storage             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Sync state schema

**File:** `data/sync_state.json`

```json
{
  "version": 1,
  "sources": {
    "github/ethereum/EIPs": {
      "files": {
        "last_sync": "2026-01-20T03:00:00Z",
        "last_commit": "abc123def456",
        "docs_synced": 892
      },
      "issues": {
        "last_sync": "2026-01-20T03:00:00Z",
        "last_issue_updated": "2026-01-19T22:15:00Z",
        "docs_synced": 5234
      }
    },
    "github/ethereum/consensus-specs": {
      "files": {
        "last_sync": "2026-01-20T03:00:00Z",
        "last_commit": "def789abc012"
      }
    },
    "arxiv": {
      "last_sync": "2026-01-15T03:00:00Z",
      "last_search_date": "2026-01-15",
      "papers_synced": 48,
      "cursor": null
    },
    "ethresearch": {
      "last_sync": "2026-01-20T03:00:00Z",
      "last_topic_id": 18234,
      "last_topic_updated": "2026-01-19T18:30:00Z",
      "topics_synced": 50
    },
    "magicians": {
      "last_sync": "2026-01-20T03:00:00Z",
      "last_topic_id": 22156,
      "error_count": 3,
      "last_error": "403 Forbidden",
      "backoff_until": "2026-01-21T03:00:00Z"
    }
  },
  "global": {
    "last_full_sync": "2026-01-01T00:00:00Z",
    "total_documents": 15426,
    "total_chunks": 89234
  }
}
```

---

## Source-specific strategies

### 1. GitHub repositories

**Update patterns:**
- Spec files: Change with merges (daily check sufficient)
- Issues/PRs: High activity (hourly for EIPs, daily for specs)

**Incremental sync:**

```python
class GitHubSyncer:
    """Incremental GitHub sync using LlamaIndex readers."""

    async def sync_files(self, repo: str, state: dict) -> SyncResult:
        """Sync repository files if commits changed."""
        last_commit = state.get("last_commit")
        current_commit = await self._get_head_commit(repo)

        if current_commit == last_commit:
            return SyncResult(changed=False)

        # Fetch all files (git is efficient with shallow clone)
        documents = self.loader.load_repository_files(repo)

        return SyncResult(
            changed=True,
            documents=documents,
            new_cursor={"last_commit": current_commit},
        )

    async def sync_issues(self, repo: str, state: dict) -> SyncResult:
        """Sync issues updated since last sync."""
        last_updated = state.get("last_issue_updated")

        # GitHub API: GET /repos/{owner}/{repo}/issues?since={timestamp}&state=all
        # Returns issues updated after timestamp
        issues = await self._fetch_issues_since(repo, since=last_updated)

        if not issues:
            return SyncResult(changed=False)

        documents = self._convert_to_documents(issues)
        newest_updated = max(i["updated_at"] for i in issues)

        return SyncResult(
            changed=True,
            documents=documents,
            new_cursor={"last_issue_updated": newest_updated},
        )
```

**Real-time option (webhooks):**

```python
# Webhook events to handle:
# - push: Re-sync changed files
# - issues: Re-ingest single issue
# - pull_request: Re-ingest single PR
# - issue_comment: Update issue content

@router.post("/webhooks/github/{repo}")
async def github_webhook(repo: str, request: Request):
    event = request.headers.get("X-GitHub-Event")
    payload = await request.json()

    if event == "push":
        # Queue file sync for changed paths
        changed = extract_changed_files(payload)
        await queue_file_sync(repo, changed)

    elif event in ("issues", "pull_request"):
        # Re-ingest single item
        number = payload["number"]
        await queue_issue_sync(repo, number)
```

---

### 2. arXiv papers

**Update pattern:** New papers published continuously, but weekly batch is sufficient.

**Incremental sync:**

```python
class ArxivSyncer:
    """Weekly arXiv paper sync."""

    SEARCH_QUERIES = [
        "ethereum",
        "smart contract",
        "blockchain consensus",
        "proof of stake",
        "EVM",
        "rollup",
        "data availability",
    ]

    async def sync(self, state: dict) -> SyncResult:
        """Fetch papers published since last sync."""
        last_search_date = state.get("last_search_date")

        # arXiv API: submittedDate:[YYYYMMDD TO YYYYMMDD]
        today = datetime.now().strftime("%Y%m%d")
        date_range = f"[{last_search_date} TO {today}]"

        new_papers = []
        for query in self.SEARCH_QUERIES:
            full_query = f"all:{query} AND submittedDate:{date_range}"
            papers = self.fetcher.search(full_query, max_results=50)
            new_papers.extend(papers)

        # Deduplicate by arxiv_id
        unique_papers = {p.arxiv_id: p for p in new_papers}.values()

        # Fetch PDFs and extract content
        documents = []
        for paper in unique_papers:
            if self.cache.has("arxiv", paper.arxiv_id):
                continue  # Already have this paper

            pdf_path = self.fetcher.fetch_pdf_cached(paper)
            if pdf_path:
                content = self.extractor.extract(pdf_path)
                documents.append(self._to_document(paper, content))

        return SyncResult(
            changed=len(documents) > 0,
            documents=documents,
            new_cursor={"last_search_date": today},
        )
```

**Considerations:**
- arXiv rate limit: 1 request per 3 seconds
- PDF downloads can be slow - use caching
- Quality scoring to filter low-quality extractions

---

### 3. Discourse forums (Ethresearch, Magicians)

**Update pattern:** New topics and replies daily.

**Incremental sync:**

```python
class DiscourseSyncer:
    """Incremental Discourse forum sync."""

    def __init__(self, base_url: str, source_name: str):
        self.base_url = base_url
        self.source_name = source_name
        self.client = DiscourseClient(base_url)

    async def sync(self, state: dict) -> SyncResult:
        """Fetch topics updated since last sync."""
        last_topic_updated = state.get("last_topic_updated")

        # Discourse /latest.json returns topics sorted by activity
        # Fetch pages until we hit topics older than last_sync
        new_topics = []
        page = 0

        async with self.client:
            while True:
                topics = await self.client.get_latest_topics(page=page)
                if not topics:
                    break

                for topic in topics:
                    # Stop when we hit already-synced topics
                    if last_topic_updated and topic.last_posted_at <= last_topic_updated:
                        break

                    # Fetch full topic with posts
                    full_topic = await self.loader.load_topic_with_posts_cached(
                        topic.topic_id
                    )
                    if full_topic:
                        new_topics.append(full_topic)

                page += 1
                if page > 10:  # Safety limit
                    break

        if not new_topics:
            return SyncResult(changed=False)

        documents = [self._to_document(t) for t in new_topics]
        newest_updated = max(t.last_posted_at for t in new_topics)

        return SyncResult(
            changed=True,
            documents=documents,
            new_cursor={"last_topic_updated": newest_updated.isoformat()},
        )

    async def sync_with_backoff(self, state: dict) -> SyncResult:
        """Sync with exponential backoff for rate limits."""
        backoff_until = state.get("backoff_until")
        if backoff_until and datetime.now() < datetime.fromisoformat(backoff_until):
            logger.info("skipping_due_to_backoff", source=self.source_name)
            return SyncResult(changed=False, skipped=True)

        try:
            return await self.sync(state)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 403):
                # Exponential backoff
                error_count = state.get("error_count", 0) + 1
                backoff_hours = min(2 ** error_count, 168)  # Max 1 week

                return SyncResult(
                    changed=False,
                    error=str(e),
                    new_cursor={
                        "error_count": error_count,
                        "last_error": str(e),
                        "backoff_until": (
                            datetime.now() + timedelta(hours=backoff_hours)
                        ).isoformat(),
                    },
                )
            raise
```

**Considerations:**
- Magicians has shown 403 errors - need graceful degradation
- Rate limit: ~1 req/sec without auth
- Consider auth tokens for higher limits

---

### 4. ACD transcripts (via GitHub pm repo)

**Update pattern:** New transcripts after each ACD call (~weekly).

**Incremental sync:**

```python
class ACDSyncer:
    """Sync ACD transcripts from ethereum/pm repo."""

    async def sync(self, state: dict) -> SyncResult:
        """Check for new transcripts via git."""
        last_commit = state.get("last_commit")

        # Use GitHub API to check for new commits
        current_commit = await self._get_head_commit("ethereum/pm")
        if current_commit == last_commit:
            return SyncResult(changed=False)

        # Get list of changed files
        changed_files = await self._get_changed_files(
            "ethereum/pm",
            base=last_commit,
            head=current_commit,
        )

        # Filter to transcript files
        transcript_files = [
            f for f in changed_files
            if f.startswith(("AllCoreDevs-EL-Meetings/", "AllCoreDevs-CL-Meetings/"))
            and f.endswith(".md")
        ]

        if not transcript_files:
            return SyncResult(
                changed=False,
                new_cursor={"last_commit": current_commit},
            )

        # Load only changed transcripts
        documents = []
        for file_path in transcript_files:
            transcript = self.loader.load_transcript(file_path)
            if transcript:
                documents.append(self._to_document(transcript))

        return SyncResult(
            changed=True,
            documents=documents,
            new_cursor={"last_commit": current_commit},
        )
```

---

## Unified orchestrator

**File:** `src/ingestion/orchestrator.py`

```python
"""Ingestion orchestrator - coordinates all source syncs."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger()


@dataclass
class SyncResult:
    """Result of a sync operation."""
    changed: bool
    documents: list = None
    new_cursor: dict = None
    error: str | None = None
    skipped: bool = False


class Syncer(Protocol):
    """Protocol for source syncers."""

    async def sync(self, state: dict) -> SyncResult: ...


@dataclass
class SyncSchedule:
    """Schedule for a sync job."""
    source: str
    syncer: Syncer
    interval: timedelta
    enabled: bool = True


class IngestionOrchestrator:
    """Orchestrate incremental syncs across all sources."""

    def __init__(
        self,
        state_file: Path = Path("data/sync_state.json"),
        schedules: list[SyncSchedule] | None = None,
    ):
        self.state_file = state_file
        self.state = self._load_state()
        self.schedules = schedules or []
        self._running = False

    def _load_state(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"version": 1, "sources": {}, "global": {}}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def get_source_state(self, source: str) -> dict:
        return self.state["sources"].get(source, {})

    def update_source_state(self, source: str, updates: dict):
        if source not in self.state["sources"]:
            self.state["sources"][source] = {}
        self.state["sources"][source].update(updates)
        self.state["sources"][source]["last_sync"] = datetime.utcnow().isoformat()
        self._save_state()

    async def sync_source(self, schedule: SyncSchedule) -> SyncResult:
        """Run sync for a single source."""
        source = schedule.source
        logger.info("starting_sync", source=source)

        state = self.get_source_state(source)
        start_time = datetime.utcnow()

        try:
            result = await schedule.syncer.sync(state)

            if result.new_cursor:
                self.update_source_state(source, result.new_cursor)

            if result.changed and result.documents:
                await self._process_documents(result.documents)
                self.update_source_state(source, {
                    "docs_synced": len(result.documents),
                })

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "sync_complete",
                source=source,
                changed=result.changed,
                documents=len(result.documents or []),
                duration_sec=round(duration, 2),
            )

            return result

        except Exception as e:
            logger.error("sync_failed", source=source, error=str(e))
            self.update_source_state(source, {
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1,
            })
            return SyncResult(changed=False, error=str(e))

    async def _process_documents(self, documents: list):
        """Process documents through the pipeline."""
        # Chunking → Embedding → Storage
        # (Delegate to existing pipeline)
        pass

    async def run_all(self, force: bool = False):
        """Run all scheduled syncs."""
        for schedule in self.schedules:
            if not schedule.enabled:
                continue

            state = self.get_source_state(schedule.source)
            last_sync = state.get("last_sync")

            # Check if due
            if not force and last_sync:
                last_sync_dt = datetime.fromisoformat(last_sync)
                next_sync = last_sync_dt + schedule.interval
                if datetime.utcnow() < next_sync:
                    logger.debug("skipping_not_due", source=schedule.source)
                    continue

            await self.sync_source(schedule)

    async def run_continuous(self, check_interval: int = 300):
        """Run continuously, checking schedules every interval."""
        self._running = True
        logger.info("starting_continuous_sync", check_interval=check_interval)

        while self._running:
            await self.run_all()
            await asyncio.sleep(check_interval)

    def stop(self):
        """Stop continuous sync."""
        self._running = False


# Default schedules
def get_default_schedules() -> list[SyncSchedule]:
    """Get default sync schedules for all sources."""
    return [
        # GitHub repos - daily for files
        SyncSchedule(
            source="github/ethereum/EIPs/files",
            syncer=GitHubFileSyncer("ethereum", "EIPs"),
            interval=timedelta(hours=24),
        ),
        # GitHub issues - more frequent
        SyncSchedule(
            source="github/ethereum/EIPs/issues",
            syncer=GitHubIssuesSyncer("ethereum", "EIPs"),
            interval=timedelta(hours=6),
        ),
        SyncSchedule(
            source="github/ethereum/ERCs/files",
            syncer=GitHubFileSyncer("ethereum", "ERCs"),
            interval=timedelta(hours=24),
        ),
        SyncSchedule(
            source="github/ethereum/ERCs/issues",
            syncer=GitHubIssuesSyncer("ethereum", "ERCs"),
            interval=timedelta(hours=6),
        ),
        # Specs - daily
        SyncSchedule(
            source="github/ethereum/consensus-specs",
            syncer=GitHubFileSyncer("ethereum", "consensus-specs"),
            interval=timedelta(hours=24),
        ),
        SyncSchedule(
            source="github/ethereum/execution-specs",
            syncer=GitHubFileSyncer("ethereum", "execution-specs"),
            interval=timedelta(hours=24),
        ),
        # ACD transcripts - weekly
        SyncSchedule(
            source="github/ethereum/pm",
            syncer=ACDSyncer(),
            interval=timedelta(days=7),
        ),
        # arXiv - weekly
        SyncSchedule(
            source="arxiv",
            syncer=ArxivSyncer(),
            interval=timedelta(days=7),
        ),
        # Forums - daily
        SyncSchedule(
            source="ethresearch",
            syncer=DiscourseSyncer("https://ethresear.ch", "ethresearch"),
            interval=timedelta(hours=24),
        ),
        SyncSchedule(
            source="magicians",
            syncer=DiscourseSyncer("https://ethereum-magicians.org", "magicians"),
            interval=timedelta(hours=24),
            enabled=False,  # Disabled due to 403 errors
        ),
    ]
```

---

## CLI interface

**File:** `scripts/sync.py`

```python
#!/usr/bin/env python3
"""Continuous ingestion CLI.

Usage:
    # Run all due syncs
    uv run python scripts/sync.py

    # Run specific source
    uv run python scripts/sync.py --source github/ethereum/EIPs/issues

    # Force sync (ignore schedule)
    uv run python scripts/sync.py --force

    # Run continuously (daemon mode)
    uv run python scripts/sync.py --daemon

    # Show sync status
    uv run python scripts/sync.py --status
"""

import argparse
import asyncio

from src.ingestion.orchestrator import (
    IngestionOrchestrator,
    get_default_schedules,
)


def show_status(orchestrator: IngestionOrchestrator):
    """Display sync status for all sources."""
    print("\nSync Status")
    print("=" * 70)

    for schedule in orchestrator.schedules:
        source = schedule.source
        state = orchestrator.get_source_state(source)

        last_sync = state.get("last_sync", "Never")
        docs = state.get("docs_synced", 0)
        error = state.get("last_error")
        enabled = "✓" if schedule.enabled else "✗"

        print(f"\n{enabled} {source}")
        print(f"  Last sync: {last_sync}")
        print(f"  Docs synced: {docs}")
        print(f"  Interval: {schedule.interval}")
        if error:
            print(f"  Last error: {error}")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Continuous ingestion")
    parser.add_argument("--source", help="Sync specific source only")
    parser.add_argument("--force", action="store_true", help="Ignore schedule")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (daemon)")

    args = parser.parse_args()

    orchestrator = IngestionOrchestrator(schedules=get_default_schedules())

    if args.status:
        show_status(orchestrator)
        return

    if args.source:
        # Find and run specific schedule
        for schedule in orchestrator.schedules:
            if schedule.source == args.source:
                await orchestrator.sync_source(schedule)
                return
        print(f"Unknown source: {args.source}")
        return

    if args.daemon:
        await orchestrator.run_continuous(check_interval=args.interval)
    else:
        await orchestrator.run_all(force=args.force)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Deployment options

### Option 1: Systemd timer (recommended for single server)

```ini
# /etc/systemd/system/eth-protocol-sync.service
[Unit]
Description=Ethereum Protocol Expert Sync
After=network.target postgresql.service

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/eth-protocol-expert
ExecStart=/home/ubuntu/.local/bin/uv run python scripts/sync.py
StandardOutput=append:/var/log/eth-protocol-sync.log
StandardError=append:/var/log/eth-protocol-sync.log
Environment="PATH=/home/ubuntu/.local/bin:/usr/bin"

# /etc/systemd/system/eth-protocol-sync.timer
[Unit]
Description=Run Ethereum Protocol Expert sync

[Timer]
# Every 6 hours
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable eth-protocol-sync.timer
sudo systemctl start eth-protocol-sync.timer
sudo systemctl status eth-protocol-sync.timer
```

### Option 2: Docker with cron

```dockerfile
# Dockerfile.sync
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install uv && uv sync

# Run sync every 6 hours
RUN echo "0 */6 * * * cd /app && uv run python scripts/sync.py >> /var/log/sync.log 2>&1" | crontab -

CMD ["cron", "-f"]
```

### Option 3: Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: eth-protocol-sync
spec:
  schedule: "0 */6 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: sync
            image: eth-protocol-expert:latest
            command: ["uv", "run", "python", "scripts/sync.py"]
            env:
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: eth-protocol-secrets
                  key: github-token
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: eth-protocol-secrets
                  key: database-url
          restartPolicy: OnFailure
```

### Option 4: Daemon mode with supervisor

```ini
# /etc/supervisor/conf.d/eth-protocol-sync.conf
[program:eth-protocol-sync]
command=/home/ubuntu/.local/bin/uv run python scripts/sync.py --daemon
directory=/home/ubuntu/eth-protocol-expert
user=ubuntu
autostart=true
autorestart=true
stderr_logfile=/var/log/eth-protocol-sync.err.log
stdout_logfile=/var/log/eth-protocol-sync.out.log
environment=PATH="/home/ubuntu/.local/bin:/usr/bin"
```

---

## Real-time updates (webhooks)

For sources that support webhooks, add real-time ingestion.

**File:** `src/api/webhooks.py`

```python
"""Webhook handlers for real-time updates."""

import hashlib
import hmac
import os
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request

from src.ingestion.orchestrator import IngestionOrchestrator

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")


def verify_github_signature(body: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature."""
    if not GITHUB_WEBHOOK_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
):
    """Handle GitHub webhook events for real-time updates."""
    body = await request.body()

    if not verify_github_signature(body, x_hub_signature_256 or ""):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    repo = payload.get("repository", {}).get("full_name")

    if x_github_event == "push":
        # Queue file sync
        changed_files = []
        for commit in payload.get("commits", []):
            changed_files.extend(commit.get("added", []))
            changed_files.extend(commit.get("modified", []))
        await queue_sync("github_files", repo=repo, files=changed_files)

    elif x_github_event in ("issues", "pull_request"):
        # Queue single issue/PR sync
        number = payload.get("issue", payload.get("pull_request", {})).get("number")
        await queue_sync("github_issue", repo=repo, number=number)

    elif x_github_event == "issue_comment":
        # Re-sync the parent issue
        issue_number = payload.get("issue", {}).get("number")
        await queue_sync("github_issue", repo=repo, number=issue_number)

    return {"status": "queued", "event": x_github_event}


async def queue_sync(sync_type: str, **kwargs):
    """Queue a sync job for async processing."""
    # In production, use a task queue (Celery, RQ, etc.)
    # For simplicity, process inline
    orchestrator = IngestionOrchestrator()
    # ... trigger appropriate syncer
```

**Webhook setup for ethereum repos:**

1. Go to each repo's Settings → Webhooks
2. Add webhook:
   - URL: `https://your-domain.com/webhooks/github`
   - Content type: `application/json`
   - Secret: Generate and store in `GITHUB_WEBHOOK_SECRET`
   - Events: Push, Issues, Pull requests, Issue comments

---

## Monitoring and alerting

```python
# src/ingestion/metrics.py

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SyncMetrics:
    """Metrics for monitoring sync health."""

    source: str
    last_success: datetime | None
    last_failure: datetime | None
    success_count_24h: int
    failure_count_24h: int
    avg_duration_sec: float
    docs_synced_24h: int


def check_sync_health(state: dict) -> list[str]:
    """Check for sync health issues."""
    alerts = []

    for source, source_state in state.get("sources", {}).items():
        last_sync = source_state.get("last_sync")
        error_count = source_state.get("error_count", 0)

        # Alert if no sync in 48 hours
        if last_sync:
            last_sync_dt = datetime.fromisoformat(last_sync)
            if datetime.utcnow() - last_sync_dt > timedelta(hours=48):
                alerts.append(f"WARN: {source} not synced in 48+ hours")

        # Alert on repeated failures
        if error_count >= 3:
            alerts.append(f"ERROR: {source} has {error_count} consecutive failures")

    return alerts
```

---

## Summary

| Source | Sync Strategy | Interval | Real-time |
|--------|---------------|----------|-----------|
| GitHub files | Commit diff | 24h | Webhook (push) |
| GitHub issues | `since` param | 6h | Webhook (issues) |
| arXiv | Date range search | 7d | ❌ |
| Ethresearch | Latest topics | 24h | ❌ |
| Magicians | Latest topics | 24h (disabled) | ❌ |
| ACD transcripts | Git diff | 7d | Webhook (push) |

## Implementation order

1. [ ] Create `src/ingestion/orchestrator.py`
2. [ ] Create source-specific syncers:
   - [ ] `GitHubFileSyncer`
   - [ ] `GitHubIssuesSyncer`
   - [ ] `ArxivSyncer`
   - [ ] `DiscourseSyncer`
   - [ ] `ACDSyncer`
3. [ ] Create `scripts/sync.py` CLI
4. [ ] Set up systemd timer
5. [ ] (Optional) Add webhook handlers
6. [ ] Add monitoring/alerting
7. [ ] Test incremental sync for each source
