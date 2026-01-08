"""Ethereum Magicians Loader - Load topics and posts from ethereum-magicians.org."""

from collections.abc import AsyncIterator

import structlog

from .discourse_client import (
    DiscourseClient,
    DiscoursePost,
    DiscourseTopic,
)
from .ethresearch_loader import LoadedForumPost, LoadedForumTopic

logger = structlog.get_logger()

MAGICIANS_BASE_URL = "https://ethereum-magicians.org"


class MagiciansLoader:
    """Load topics and posts from ethereum-magicians.org.

    Ethereum Magicians is the forum for EIP discussion and governance.
    It's where many EIPs are proposed, debated, and refined. Runs on Discourse.
    """

    def __init__(
        self,
        base_url: str = MAGICIANS_BASE_URL,
        rate_limit_delay: float = 1.0,
    ):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.source = "magicians"

    def _make_client(self) -> DiscourseClient:
        return DiscourseClient(
            base_url=self.base_url,
            rate_limit_delay=self.rate_limit_delay,
        )

    async def load_latest_topics(
        self, max_pages: int = 10
    ) -> list[DiscourseTopic]:
        """Load recent topics from Ethereum Magicians."""
        async with self._make_client() as client:
            topics = []
            async for topic in client.iter_all_topics(max_pages=max_pages):
                topics.append(topic)
            logger.info("loaded_magicians_topics", count=len(topics))
            return topics

    async def search_eip_discussions(self, eip_number: int) -> list[DiscourseTopic]:
        """Search for discussions about a specific EIP.

        Ethereum Magicians often has official EIP discussion threads
        with titles like 'EIP-XXXX: Title'.
        """
        async with self._make_client() as client:
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
                "searched_magicians_eip",
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

            # Fetch raw content for posts that don't have it
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

    async def load_eip_canonical_discussion(
        self, eip_number: int
    ) -> LoadedForumTopic | None:
        """Try to find and load the canonical EIP discussion thread.

        Ethereum Magicians typically has an official discussion thread for each EIP
        with a title pattern like 'EIP-XXXX: <title>'.
        """
        async with self._make_client() as client:
            topics = await client.search_topics(f"EIP-{eip_number}:")

            # Find the most likely canonical thread
            # Prefer exact title match, then highest engagement
            canonical = None
            for topic in topics:
                if topic.title.lower().startswith(f"eip-{eip_number}:"):
                    if canonical is None or topic.posts_count > canonical.posts_count:
                        canonical = topic

            if canonical:
                return await self.load_topic_with_posts(canonical.topic_id)

            # Fallback: any topic mentioning the EIP
            if topics:
                return await self.load_topic_with_posts(topics[0].topic_id)

            return None

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

        for eip_number in eip_numbers:
            topics = await self.search_eip_discussions(eip_number)

            loaded_topics = []
            for topic in topics[:10]:  # Limit per EIP
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
