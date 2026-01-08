"""Ethresear.ch Loader - Load topics and posts from ethresear.ch (Discourse)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import structlog

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
    ):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.source = "ethresearch"

    def _make_client(self) -> DiscourseClient:
        return DiscourseClient(
            base_url=self.base_url,
            rate_limit_delay=self.rate_limit_delay,
        )

    async def load_latest_topics(
        self, max_pages: int = 10
    ) -> list[DiscourseTopic]:
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

    async def load_topic_with_posts(
        self, topic_id: int
    ) -> LoadedForumTopic | None:
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

    def _to_loaded_post(
        self, post: DiscoursePost, topic: DiscourseTopic
    ) -> LoadedForumPost:
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
