"""Ingestion Orchestrator - Manage sync state and coordinate incremental updates."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class SourceSyncState:
    """Sync state for a single source."""

    source_name: str
    last_sync: str | None = None
    last_commit: str | None = None
    last_cursor: str | None = None
    docs_synced: int = 0
    errors: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SyncState:
    """Global sync state across all sources."""

    version: str = "1.0"
    last_updated: str | None = None
    sources: dict[str, SourceSyncState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "sources": {name: asdict(state) for name, state in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyncState":
        """Load from dict."""
        state = cls(
            version=data.get("version", "1.0"),
            last_updated=data.get("last_updated"),
        )
        for name, source_data in data.get("sources", {}).items():
            state.sources[name] = SourceSyncState(**source_data)
        return state


class IngestionOrchestrator:
    """Orchestrate incremental ingestion across multiple sources."""

    def __init__(self, state_path: str | Path = "data/sync_state.json"):
        self.state_path = Path(state_path)
        self.state: SyncState | None = None

    def load_state(self) -> SyncState:
        """Load sync state from disk."""
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.state = SyncState.from_dict(data)
            logger.info("loaded_sync_state", sources=len(self.state.sources))
        else:
            self.state = SyncState()
            logger.info("created_new_sync_state")
        return self.state

    def save_state(self) -> None:
        """Save sync state to disk."""
        if self.state is None:
            return
        self.state.last_updated = datetime.now(UTC).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        logger.info("saved_sync_state", path=str(self.state_path))

    def get_source_state(self, source_name: str) -> SourceSyncState:
        """Get or create state for a source."""
        if self.state is None:
            self.load_state()
        if source_name not in self.state.sources:
            self.state.sources[source_name] = SourceSyncState(source_name=source_name)
        return self.state.sources[source_name]

    def update_source_state(
        self,
        source_name: str,
        *,
        commit: str | None = None,
        cursor: str | None = None,
        docs_synced: int = 0,
        errors: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Update state for a source after sync."""
        state = self.get_source_state(source_name)
        state.last_sync = datetime.now(UTC).isoformat()
        if commit is not None:
            state.last_commit = commit
        if cursor is not None:
            state.last_cursor = cursor
        state.docs_synced += docs_synced
        state.errors += errors
        if metadata:
            state.metadata.update(metadata)
        self.save_state()

    def needs_sync(self, source_name: str, current_commit: str | None = None) -> bool:
        """Check if a source needs syncing."""
        state = self.get_source_state(source_name)
        if state.last_sync is None:
            return True
        if current_commit and state.last_commit != current_commit:
            return True
        return False


class GitHubSyncer:
    """Sync git-based repositories."""

    def __init__(self, orchestrator: IngestionOrchestrator):
        self.orchestrator = orchestrator

    async def sync_repo(
        self,
        source_name: str,
        repo_path: Path,
        ingest_func,
    ) -> dict:
        """Sync a git repository and run ingestion if needed."""
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        current_commit = result.stdout.strip() if result.returncode == 0 else None

        state = self.orchestrator.get_source_state(source_name)

        if not self.orchestrator.needs_sync(source_name, current_commit):
            logger.info(
                "source_already_up_to_date",
                source=source_name,
                commit=current_commit[:8] if current_commit else None,
            )
            return {"synced": 0, "status": "already_up_to_date"}

        logger.info(
            "syncing_source",
            source=source_name,
            from_commit=state.last_commit[:8] if state.last_commit else None,
            to_commit=current_commit[:8] if current_commit else None,
        )

        try:
            docs_synced = await ingest_func()
            self.orchestrator.update_source_state(
                source_name,
                commit=current_commit,
                docs_synced=docs_synced,
            )
            return {"synced": docs_synced, "status": "success"}
        except Exception as e:
            logger.error("sync_failed", source=source_name, error=str(e))
            self.orchestrator.update_source_state(source_name, errors=1)
            return {"synced": 0, "status": "error", "error": str(e)}


class ArxivSyncer:
    """Sync arXiv papers with date-based incremental fetching."""

    def __init__(self, orchestrator: IngestionOrchestrator):
        self.orchestrator = orchestrator

    async def sync(self, ingest_func, max_papers: int = 50) -> dict:
        """Sync arXiv papers from last sync date."""
        state = self.orchestrator.get_source_state("arxiv")

        last_date = None
        if state.last_sync:
            last_date = state.last_sync[:10]

        logger.info("syncing_arxiv", since_date=last_date, max_papers=max_papers)

        try:
            docs_synced = await ingest_func(since_date=last_date, max_papers=max_papers)
            self.orchestrator.update_source_state("arxiv", docs_synced=docs_synced)
            return {"synced": docs_synced, "status": "success"}
        except Exception as e:
            logger.error("arxiv_sync_failed", error=str(e))
            self.orchestrator.update_source_state("arxiv", errors=1)
            return {"synced": 0, "status": "error", "error": str(e)}


class DiscourseSyncer:
    """Sync Discourse forums with bumped_at cursor."""

    def __init__(self, orchestrator: IngestionOrchestrator):
        self.orchestrator = orchestrator

    async def sync(self, source_name: str, sync_func, ingest_func) -> dict:
        """Sync a Discourse forum incrementally."""
        state = self.orchestrator.get_source_state(source_name)

        logger.info(
            "syncing_discourse",
            source=source_name,
            last_cursor=state.last_cursor,
        )

        try:
            sync_result = await sync_func(since_cursor=state.last_cursor)
            new_cursor = sync_result.get("last_bumped_at")

            if sync_result.get("synced", 0) == 0 and sync_result.get("skipped", 0) > 0:
                logger.info("discourse_already_up_to_date", source=source_name)
                return {"synced": 0, "status": "already_up_to_date"}

            docs_synced = await ingest_func()
            self.orchestrator.update_source_state(
                source_name,
                cursor=new_cursor,
                docs_synced=docs_synced,
                metadata={"last_sync_result": sync_result},
            )
            return {"synced": docs_synced, "status": "success"}
        except Exception as e:
            logger.error("discourse_sync_failed", source=source_name, error=str(e))
            self.orchestrator.update_source_state(source_name, errors=1)
            return {"synced": 0, "status": "error", "error": str(e)}
