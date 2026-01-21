#!/usr/bin/env python3
"""Sync ethresear.ch forum topics to local cache.

This script downloads topics from ethresear.ch API and stores them in the local
cache. It does NOT ingest into the database - use ingest_ethresearch.py for that.

Features:
- Smart staleness detection: only re-fetches topics with new posts/activity
- Parallel fetching with rate limiting (configurable concurrency)
- Incremental sync: only fetch topics modified since last sync
- Statistics display: shows total topics on forum vs cached

Usage:
    uv run python scripts/sync_ethresearch.py [--max-topics 1000]
    uv run python scripts/sync_ethresearch.py --stats-only
    uv run python scripts/sync_ethresearch.py --incremental
    uv run python scripts/sync_ethresearch.py --max-topics 500 --force
    uv run python scripts/sync_ethresearch.py --max-topics 500 --max-concurrent 10
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv

from src.ingestion import EthresearchLoader, RawContentCache
from src.ingestion.discourse_client import DiscourseClient

load_dotenv()
logger = structlog.get_logger()

ETHRESEARCH_BASE_URL = "https://ethresear.ch"
SYNC_STATE_FILE = Path("data/cache/ethresearch/.sync_state.json")


def load_sync_state() -> dict:
    """Load the last sync state."""
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {}


def save_sync_state(state: dict) -> None:
    """Save the sync state."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


async def show_statistics(cache: RawContentCache) -> dict:
    """Show sync statistics: forum total vs cached."""
    async with DiscourseClient(ETHRESEARCH_BASE_URL) as client:
        stats = await client.get_statistics()

    cache_stats = cache.stats("ethresearch")
    cached_count = cache_stats["entry_count"]

    print("\n" + "=" * 60)
    print("ethresear.ch Sync Statistics")
    print("=" * 60)
    print(f"Forum total topics:     {stats.topics_count:,}")
    print(f"Cached topics:          {cached_count:,}")
    print(f"Cache coverage:         {cached_count / stats.topics_count * 100:.1f}%")
    print(f"Missing topics:         {stats.topics_count - cached_count:,}")
    print("-" * 60)
    print(f"Forum total posts:      {stats.posts_count:,}")
    print(f"Forum total users:      {stats.users_count:,}")
    print(f"New topics (7 days):    {stats.topics_7_days:,}")
    print(f"New topics (30 days):   {stats.topics_30_days:,}")
    print("-" * 60)
    print(f"Cache size:             {cache_stats['total_mb']:.1f} MB")
    print("=" * 60 + "\n")

    return {
        "forum_topics": stats.topics_count,
        "cached_topics": cached_count,
        "new_topics_7d": stats.topics_7_days,
        "new_topics_30d": stats.topics_30_days,
    }


async def sync_incremental(
    loader: EthresearchLoader,
    max_concurrent: int = 5,
) -> dict[str, int]:
    """Sync only topics modified since last sync.

    Uses the bumped_at field to efficiently find topics that have been
    updated since the last sync, stopping early when reaching old topics.
    """
    sync_state = load_sync_state()
    last_sync = sync_state.get("last_sync_at")

    if last_sync:
        since = datetime.fromisoformat(last_sync)
        logger.info("incremental_sync_start", since=last_sync)
    else:
        # First sync - fall back to full sync
        logger.info("no_previous_sync_found", msg="performing full sync")
        since = None

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {"synced": 0, "skipped": 0, "failed": 0}
    results_lock = asyncio.Lock()
    sync_start = datetime.utcnow()

    async def sync_with_semaphore(topic) -> None:
        async with semaphore:
            await asyncio.sleep(loader.rate_limit_delay)
            try:
                if not loader._is_topic_stale(
                    topic.topic_id,
                    topic.bumped_at,
                    topic.posts_count,
                ):
                    async with results_lock:
                        results["skipped"] += 1
                    return

                success = await loader.sync_topic(
                    topic.topic_id,
                    api_metadata=topic,
                    force=False,
                )
                async with results_lock:
                    if success:
                        results["synced"] += 1
                    else:
                        results["failed"] += 1
            except Exception as e:
                logger.error("sync_topic_error", topic_id=topic.topic_id, error=str(e))
                async with results_lock:
                    results["failed"] += 1

    # Collect topics that need syncing
    topics_to_sync = []
    async with DiscourseClient(ETHRESEARCH_BASE_URL) as client:
        if since:
            # Incremental: only topics modified since last sync
            async for topic in client.iter_topics_since(since):
                topics_to_sync.append(topic)
                if len(topics_to_sync) % 100 == 0:
                    logger.debug("collecting_topics", count=len(topics_to_sync))
        else:
            # Full sync: all topics
            async for topic in client.iter_all_topics():
                topics_to_sync.append(topic)
                if len(topics_to_sync) % 100 == 0:
                    logger.debug("collecting_topics", count=len(topics_to_sync))

    logger.info(
        "topics_to_check",
        count=len(topics_to_sync),
        incremental=since is not None,
    )

    # Sync topics in parallel
    tasks = [sync_with_semaphore(topic) for topic in topics_to_sync]
    await asyncio.gather(*tasks)

    # Save sync state
    save_sync_state(
        {
            "last_sync_at": sync_start.isoformat(),
            "last_sync_results": results,
        }
    )

    logger.info(
        "incremental_sync_complete",
        synced=results["synced"],
        skipped=results["skipped"],
        failed=results["failed"],
    )
    return results


async def sync_ethresearch(
    max_topics: int = 1000,
    force: bool = False,
    max_concurrent: int = 5,
    incremental: bool = False,
    stats_only: bool = False,
) -> None:
    """Sync ethresear.ch topics to local cache."""
    cache = RawContentCache()
    loader = EthresearchLoader(cache=cache)

    # Always show statistics first
    await show_statistics(cache)

    if stats_only:
        return

    if incremental:
        results = await sync_incremental(loader, max_concurrent=max_concurrent)
    else:
        # Show initial cache stats
        initial_stats = cache.stats("ethresearch")
        logger.info(
            "starting_sync",
            max_topics=max_topics,
            force=force,
            max_concurrent=max_concurrent,
            cached_entries=initial_stats["entry_count"],
        )

        # Sync topics with parallel fetching
        results = await loader.sync_latest_topics(
            max_topics=max_topics,
            force=force,
            max_concurrent=max_concurrent,
        )

    # Show final cache stats
    final_stats = cache.stats("ethresearch")
    logger.info(
        "sync_finished",
        synced=results["synced"],
        skipped=results["skipped"],
        failed=results["failed"],
        total_cached=final_stats["entry_count"],
        total_mb=final_stats["total_mb"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ethresear.ch topics to local cache")
    parser.add_argument(
        "--max-topics",
        type=int,
        default=1000,
        help="Maximum number of topics to sync (default: 1000, ignored with --incremental)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-fetch even if cached (ignores staleness check)",
    )
    parser.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        help="Only sync topics modified since last sync",
    )
    parser.add_argument(
        "--stats-only",
        "-s",
        action="store_true",
        help="Only show statistics, don't sync",
    )
    args = parser.parse_args()

    asyncio.run(
        sync_ethresearch(
            max_topics=args.max_topics,
            force=args.force,
            max_concurrent=args.max_concurrent,
            incremental=args.incremental,
            stats_only=args.stats_only,
        )
    )


if __name__ == "__main__":
    main()
