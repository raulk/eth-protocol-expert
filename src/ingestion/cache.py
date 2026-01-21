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
        return self.cache_dir / source

    def _get_index_path(self, source: str) -> Path:
        return self._get_source_dir(source) / "index.json"

    def _load_index(self, source: str) -> dict:
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
                "entries": {},
            }

        return self._indexes[source]

    def _save_index(self, source: str) -> None:
        index = self._indexes.get(source)
        if not index:
            return

        index_path = self._get_index_path(source)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def has(self, source: str, item_id: str) -> bool:
        index = self._load_index(source)
        return item_id in index["entries"]

    def get(self, source: str, item_id: str) -> CacheEntry | None:
        index = self._load_index(source)
        entry_data = index["entries"].get(item_id)
        if not entry_data:
            return None
        return CacheEntry(**entry_data)

    def get_content_path(self, source: str, item_id: str) -> Path | None:
        entry = self.get(source, item_id)
        if not entry:
            return None
        return self._get_source_dir(source) / entry.content_path

    def get_content(self, source: str, item_id: str) -> bytes | None:
        path = self.get_content_path(source, item_id)
        if not path or not path.exists():
            return None
        return path.read_bytes()

    def get_content_text(self, source: str, item_id: str) -> str | None:
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
            source: Source identifier (e.g., "arxiv", "ethresearch")
            item_id: Unique item ID within source
            content: Raw content bytes
            meta: Source-specific metadata dict
            updated_at: Last modified time at source
            etag: HTTP ETag for conditional requests
            content_subpath: Override default content path
        """
        source_dir = self._get_source_dir(source)

        if content_subpath:
            content_path = content_subpath
        else:
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
        max_age_hours: float = 24 * 7,
    ) -> bool:
        """Check if cached item is stale (missing or older than max_age_hours)."""
        entry = self.get(source, item_id)
        if not entry:
            return True

        fetched_at = datetime.fromisoformat(entry.fetched_at)
        age_hours = (datetime.utcnow() - fetched_at).total_seconds() / 3600
        return age_hours > max_age_hours

    def list_entries(self, source: str) -> list[CacheEntry]:
        index = self._load_index(source)
        return [CacheEntry(**data) for data in index["entries"].values()]

    def stats(self, source: str) -> dict:
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

    def clear_source(self, source: str) -> int:
        """Clear all cached entries for a source. Returns count of entries removed."""
        index = self._load_index(source)
        count = len(index["entries"])

        source_dir = self._get_source_dir(source)
        if source_dir.exists():
            import shutil

            shutil.rmtree(source_dir)

        self._indexes.pop(source, None)
        return count

    def detect_drift(self, source: str, item_id: str, new_content: bytes) -> bool:
        """Check if content has changed since last cache.

        Args:
            source: Source identifier
            item_id: Unique item ID
            new_content: New content to compare against cached content

        Returns:
            True if content differs from cache (drift detected), False otherwise.
            Also returns True if item is not in cache.
        """
        entry = self.get(source, item_id)
        if not entry:
            return True

        new_hash = f"sha256:{hashlib.sha256(new_content).hexdigest()}"
        drifted = entry.content_hash != new_hash

        if drifted:
            logger.info(
                "drift_detected",
                source=source,
                item_id=item_id,
                old_hash=entry.content_hash[:20],
                new_hash=new_hash[:20],
            )

        return drifted

    def get_drift_report(self, source: str, new_items: dict[str, bytes]) -> dict:
        """Generate a drift report comparing new content against cache.

        Args:
            source: Source identifier
            new_items: Dict mapping item_id to new content bytes

        Returns:
            Dict with keys: added, modified, unchanged, removed
        """
        index = self._load_index(source)
        cached_ids = set(index["entries"].keys())
        new_ids = set(new_items.keys())

        added = new_ids - cached_ids
        removed = cached_ids - new_ids
        common = new_ids & cached_ids

        modified = set()
        unchanged = set()

        for item_id in common:
            if self.detect_drift(source, item_id, new_items[item_id]):
                modified.add(item_id)
            else:
                unchanged.add(item_id)

        return {
            "added": list(added),
            "modified": list(modified),
            "unchanged": list(unchanged),
            "removed": list(removed),
            "summary": {
                "added_count": len(added),
                "modified_count": len(modified),
                "unchanged_count": len(unchanged),
                "removed_count": len(removed),
            },
        }
