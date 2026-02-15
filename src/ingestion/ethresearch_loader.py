"""Ethresear.ch Loader - Load topics and posts from ethresear.ch (Discourse)."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import structlog

from .cache import RawContentCache
from .discourse_client import (
    DiscourseClient,
    DiscoursePost,
    DiscourseTopic,
)

logger = structlog.get_logger()

ETHRESEARCH_BASE_URL = "https://ethresear.ch"


@dataclass
class LoadedForumPost:
    """A forum post loaded for ingestion into the system."""

    source: str  # "ethresearch" or "magicians"
    topic_id: int
    post_id: int
    post_number: int
    username: str
    content: str  # Raw markdown
    reply_to_post_number: int | None
    created_at: datetime
    updated_at: datetime
    topic_title: str
    category: str | None
    tags: list[str]


@dataclass
class LoadedForumTopic:
    """A forum topic with all its posts."""

    source: str
    topic_id: int
    title: str
    slug: str
    category: str | None
    tags: list[str]
    posts_count: int
    created_at: datetime
    last_posted_at: datetime | None
    bumped_at: datetime | None  # Updated on any activity (new post, edit, etc.)
    posts: list[LoadedForumPost]


class EthresearchLoader:
    """Load topics and posts from ethresear.ch.

    Ethresear.ch is the Ethereum Research forum, focusing on protocol research,
    cryptography, and theoretical work. It runs on Discourse.
    """

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

    def _make_client(self) -> DiscourseClient:
        return DiscourseClient(
            base_url=self.base_url,
            rate_limit_delay=self.rate_limit_delay,
        )

    async def load_latest_topics(self, max_pages: int = 10) -> list[DiscourseTopic]:
        """Load recent topics from ethresear.ch."""
        async with self._make_client() as client:
            topics = []
            async for topic in client.iter_all_topics(max_pages=max_pages):
                topics.append(topic)
            logger.info("loaded_ethresearch_topics", count=len(topics))
            return topics

    async def search_eip_discussions(self, eip_number: int) -> list[DiscourseTopic]:
        """Search for discussions mentioning a specific EIP."""
        async with self._make_client() as client:
            # Try multiple search patterns
            results = []
            for query in [f"EIP-{eip_number}", f"EIP {eip_number}"]:
                topics = await client.search_topics(query)
                results.extend(topics)

            # Deduplicate by topic_id
            seen = set()
            unique = []
            for topic in results:
                if topic.topic_id not in seen:
                    seen.add(topic.topic_id)
                    unique.append(topic)

            logger.info(
                "searched_ethresearch_eip",
                eip_number=eip_number,
                results=len(unique),
            )
            return unique

    async def load_topic_with_posts(self, topic_id: int) -> LoadedForumTopic | None:
        """Load a complete topic with all posts (raw markdown)."""
        async with self._make_client() as client:
            result = await client.get_topic_with_posts(topic_id)
            if not result:
                return None

            # If posts don't have raw content, fetch them individually
            posts_with_raw = []
            for post in result.posts:
                if not post.raw:
                    full_post = await client.get_post(post.post_id)
                    if full_post:
                        posts_with_raw.append(full_post)
                else:
                    posts_with_raw.append(post)

            return self._to_loaded_topic(result.topic, posts_with_raw)

    async def iter_topics_with_posts(
        self, max_topics: int = 100
    ) -> AsyncIterator[LoadedForumTopic]:
        """Iterate through topics with their full post content."""
        async with self._make_client() as client:
            topic_count = 0
            async for topic in client.iter_all_topics():
                if topic_count >= max_topics:
                    break

                loaded = await self.load_topic_with_posts(topic.topic_id)
                if loaded:
                    yield loaded
                    topic_count += 1

    def _to_loaded_post(self, post: DiscoursePost, topic: DiscourseTopic) -> LoadedForumPost:
        """Convert a Discourse post to a LoadedForumPost."""
        return LoadedForumPost(
            source=self.source,
            topic_id=post.topic_id,
            post_id=post.post_id,
            post_number=post.post_number,
            username=post.username,
            content=post.raw,
            reply_to_post_number=post.reply_to_post_number,
            created_at=post.created_at,
            updated_at=post.updated_at,
            topic_title=topic.title,
            category=topic.category_name,
            tags=topic.tags,
        )

    def _to_loaded_topic(
        self, topic: DiscourseTopic, posts: list[DiscoursePost]
    ) -> LoadedForumTopic:
        """Convert a Discourse topic with posts to a LoadedForumTopic."""
        loaded_posts = [self._to_loaded_post(p, topic) for p in posts]

        return LoadedForumTopic(
            source=self.source,
            topic_id=topic.topic_id,
            title=topic.title,
            slug=topic.slug,
            category=topic.category_name,
            tags=topic.tags,
            posts_count=topic.posts_count,
            created_at=topic.created_at,
            last_posted_at=topic.last_posted_at,
            bumped_at=topic.bumped_at,
            posts=loaded_posts,
        )

    async def load_all_eip_discussions(
        self, eip_numbers: list[int]
    ) -> dict[int, list[LoadedForumTopic]]:
        """Load discussions for multiple EIPs."""
        results: dict[int, list[LoadedForumTopic]] = {}

        async with self._make_client():
            for eip_number in eip_numbers:
                topics = await self.search_eip_discussions(eip_number)

                loaded_topics = []
                for topic in topics[:10]:  # Limit per EIP to avoid rate limiting
                    loaded = await self.load_topic_with_posts(topic.topic_id)
                    if loaded:
                        loaded_topics.append(loaded)

                results[eip_number] = loaded_topics
                logger.debug(
                    "loaded_eip_discussions",
                    eip_number=eip_number,
                    topics=len(loaded_topics),
                )

        return results

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
            "bumped_at": topic.bumped_at.isoformat() if topic.bumped_at else None,
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
            last_posted_at=datetime.fromisoformat(data["last_posted_at"])
            if data.get("last_posted_at")
            else None,
            bumped_at=datetime.fromisoformat(data["bumped_at"]) if data.get("bumped_at") else None,
            posts=posts,
        )

    # -------------------------------------------------------------------------
    # Sync methods (download to cache only, no processing)
    # -------------------------------------------------------------------------

    def _is_topic_stale(
        self,
        topic_id: int,
        api_bumped_at: datetime | None,
        api_posts_count: int,
    ) -> bool:
        """Check if cached topic is stale by comparing with API metadata.

        A topic is stale if:
        - Not in cache
        - API shows newer bumped_at than cached
        - API shows more posts than cached
        """
        item_id = f"topic-{topic_id}"
        cached_data = self.cache.get_content_text(self.source, item_id)
        if not cached_data:
            return True  # Not cached, needs fetch

        try:
            data = json.loads(cached_data)
            cached_posts_count = data.get("posts_count", 0)

            # Check if API has newer posts
            if api_posts_count > cached_posts_count:
                logger.debug(
                    "topic_has_new_posts",
                    topic_id=topic_id,
                    cached=cached_posts_count,
                    api=api_posts_count,
                )
                return True

            # Check bumped_at - this captures any activity (new post, edit, etc.)
            cached_bumped = data.get("bumped_at") or data.get("last_posted_at")
            if api_bumped_at and cached_bumped:
                cached_dt = datetime.fromisoformat(cached_bumped)
                if api_bumped_at > cached_dt:
                    logger.debug(
                        "topic_has_newer_activity",
                        topic_id=topic_id,
                        cached=cached_bumped,
                        api=api_bumped_at.isoformat(),
                    )
                    return True

            return False  # Cache is fresh
        except (json.JSONDecodeError, KeyError):
            return True  # Corrupted cache, re-fetch

    async def sync_topic(
        self,
        topic_id: int,
        api_metadata: DiscourseTopic | None = None,
        force: bool = False,
    ) -> bool:
        """Sync a single topic to cache. Returns True if synced/cached.

        Args:
            topic_id: The topic ID to sync
            api_metadata: Optional topic metadata from listing (for smart staleness check)
            force: Force re-fetch even if cached
        """
        item_id = f"topic-{topic_id}"

        # Smart staleness check using API metadata if available
        if not force:
            if api_metadata:
                if not self._is_topic_stale(
                    topic_id,
                    api_metadata.bumped_at,
                    api_metadata.posts_count,
                ):
                    logger.debug("topic_cache_fresh", topic_id=topic_id)
                    return True
            elif self.cache.has(self.source, item_id):
                # No API metadata, just check if exists (legacy behavior)
                logger.debug("topic_already_cached", topic_id=topic_id)
                return True

        # Fetch from API
        topic = await self.load_topic_with_posts(topic_id)
        if not topic:
            logger.warning("topic_not_found", topic_id=topic_id)
            return False

        # Use the listing endpoint's bumped_at if it's newer than the individual
        # topic endpoint's. The two endpoints can diverge by minutes, causing
        # the staleness check (which compares against the listing) to perpetually
        # consider the cache stale.
        topic_dict = self._topic_to_cache_dict(topic)
        if api_metadata and api_metadata.bumped_at:
            cached_bumped = topic_dict.get("bumped_at")
            if cached_bumped:
                fetched_dt = datetime.fromisoformat(cached_bumped)
                if api_metadata.bumped_at > fetched_dt:
                    topic_dict["bumped_at"] = api_metadata.bumped_at.isoformat()

        self.cache.put(
            source=self.source,
            item_id=item_id,
            content=json.dumps(topic_dict, indent=2).encode(),
            meta={"title": topic.title, "posts_count": len(topic.posts)},
            updated_at=topic.last_posted_at,
            content_subpath=f"topics/{topic_id}.json",
        )
        logger.debug("synced_topic", topic_id=topic_id, title=topic.title[:50])
        return True

    async def sync_latest_topics(
        self,
        max_topics: int = 100,
        force: bool = False,
        max_concurrent: int = 5,
    ) -> dict[str, int]:
        """Sync latest topics to cache with parallel fetching.

        Args:
            max_topics: Maximum number of topics to sync
            force: Force re-fetch even if cached
            max_concurrent: Maximum concurrent API requests (rate limiting)

        Returns:
            Dict with counts: synced, skipped, failed
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {"synced": 0, "skipped": 0, "failed": 0}
        results_lock = asyncio.Lock()

        async def sync_with_semaphore(topic: DiscourseTopic) -> None:
            async with semaphore:
                # Add small delay between requests for rate limiting
                await asyncio.sleep(self.rate_limit_delay)

                try:
                    if not force and not self._is_topic_stale(
                        topic.topic_id,
                        topic.bumped_at,
                        topic.posts_count,
                    ):
                        async with results_lock:
                            results["skipped"] += 1
                        return

                    success = await self.sync_topic(
                        topic.topic_id,
                        api_metadata=topic,
                        force=force,
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

        # Collect topics first (can't parallelize the listing itself)
        topics_to_sync: list[DiscourseTopic] = []
        async with self._make_client() as client:
            async for topic in client.iter_all_topics():
                if len(topics_to_sync) >= max_topics:
                    break
                topics_to_sync.append(topic)

        logger.info("collected_topics_to_sync", count=len(topics_to_sync))

        # Sync topics in parallel with semaphore rate limiting
        tasks = [sync_with_semaphore(topic) for topic in topics_to_sync]
        await asyncio.gather(*tasks)

        logger.info(
            "sync_complete",
            synced=results["synced"],
            skipped=results["skipped"],
            failed=results["failed"],
        )
        return results

    # -------------------------------------------------------------------------
    # Load methods (read from cache only, no API calls)
    # -------------------------------------------------------------------------

    def load_topic_from_cache(self, topic_id: int) -> LoadedForumTopic | None:
        """Load a topic from cache only. Returns None if not cached."""
        item_id = f"topic-{topic_id}"
        cached_data = self.cache.get_content_text(self.source, item_id)
        if not cached_data:
            return None
        return self._cache_dict_to_topic(json.loads(cached_data))

    def iter_topics_from_cache(self) -> list[LoadedForumTopic]:
        """Load all cached topics. No API calls."""
        topics = []
        entries = self.cache.list_entries(self.source)
        for entry in entries:
            if entry.item_id.startswith("topic-"):
                cached_data = self.cache.get_content_text(self.source, entry.item_id)
                if cached_data:
                    topics.append(self._cache_dict_to_topic(json.loads(cached_data)))
        logger.info("loaded_topics_from_cache", count=len(topics))
        return topics

    def get_cached_topic_ids(self) -> list[int]:
        """Get list of topic IDs in cache."""
        entries = self.cache.list_entries(self.source)
        topic_ids = []
        for entry in entries:
            if entry.item_id.startswith("topic-"):
                topic_ids.append(int(entry.item_id.replace("topic-", "")))
        return topic_ids

    # -------------------------------------------------------------------------
    # Legacy combined methods (for backwards compatibility)
    # -------------------------------------------------------------------------

    async def load_topic_with_posts_cached(
        self,
        topic_id: int,
        max_age_hours: float = 24 * 7,
    ) -> LoadedForumTopic | None:
        """Load topic with caching (combined sync+load)."""
        item_id = f"topic-{topic_id}"

        # Check cache first
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
        """Iterate through topics with caching (combined sync+load)."""
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
