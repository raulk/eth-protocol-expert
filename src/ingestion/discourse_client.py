"""Discourse API Client - Generic client for Discourse forum APIs."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class DiscoursePost:
    """A single post from a Discourse forum."""
    post_id: int
    topic_id: int
    post_number: int
    username: str
    raw: str  # Original markdown content
    reply_to_post_number: int | None
    created_at: datetime
    updated_at: datetime


@dataclass
class DiscourseTopic:
    """A topic (thread) from a Discourse forum."""
    topic_id: int
    title: str
    category_id: int | None
    category_name: str | None
    slug: str
    posts_count: int
    created_at: datetime
    last_posted_at: datetime | None
    tags: list[str]


@dataclass
class DiscourseTopicWithPosts:
    """A topic with all its posts loaded."""
    topic: DiscourseTopic
    posts: list[DiscoursePost]


class DiscourseClient:
    """Async client for Discourse forum APIs with rate limiting.

    Discourse provides a JSON API at standard endpoints. This client handles:
    - Rate limiting (~1 req/sec safe without auth)
    - Pagination for topics and posts
    - Fetching raw markdown (not cooked HTML)
    """

    def __init__(
        self,
        base_url: str,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time: float = 0.0
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "DiscourseClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a GET request with rate limiting and retries."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with.")

        await self._rate_limit()

        for attempt in range(self.max_retries):
            try:
                response = await self._client.get(endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - back off exponentially
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    logger.warning(
                        "rate_limited",
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:
                    # Server error - retry
                    logger.warning(
                        "server_error",
                        endpoint=endpoint,
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                else:
                    raise
            except httpx.RequestError as e:
                logger.warning(
                    "request_error",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                )
                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))

        raise RuntimeError(f"Failed to fetch {endpoint} after {self.max_retries} attempts")

    async def get_latest_topics(self, page: int = 0) -> list[DiscourseTopic]:
        """Fetch latest topics from the forum."""
        data = await self._get("/latest.json", params={"page": page})
        topics = []

        topic_list = data.get("topic_list", {})
        categories = {c["id"]: c["name"] for c in data.get("categories", [])}

        for t in topic_list.get("topics", []):
            topics.append(DiscourseTopic(
                topic_id=t["id"],
                title=t["title"],
                category_id=t.get("category_id"),
                category_name=categories.get(t.get("category_id")),
                slug=t["slug"],
                posts_count=t["posts_count"],
                created_at=_parse_datetime(t["created_at"]),
                last_posted_at=_parse_datetime(t.get("last_posted_at")),
                tags=t.get("tags", []),
            ))

        logger.debug("fetched_latest_topics", page=page, count=len(topics))
        return topics

    async def iter_all_topics(self, max_pages: int = 100) -> AsyncIterator[DiscourseTopic]:
        """Iterate through all topics across pages."""
        for page in range(max_pages):
            topics = await self.get_latest_topics(page=page)
            if not topics:
                break
            for topic in topics:
                yield topic

    async def search_topics(self, query: str) -> list[DiscourseTopic]:
        """Search for topics matching a query (e.g., 'EIP-4844')."""
        data = await self._get("/search.json", params={"q": query})
        topics = []

        for t in data.get("topics", []):
            topics.append(DiscourseTopic(
                topic_id=t["id"],
                title=t["title"],
                category_id=t.get("category_id"),
                category_name=None,  # Not included in search results
                slug=t["slug"],
                posts_count=t.get("posts_count", 1),
                created_at=_parse_datetime(t["created_at"]),
                last_posted_at=_parse_datetime(t.get("last_posted_at")),
                tags=t.get("tags", []),
            ))

        logger.debug("searched_topics", query=query, count=len(topics))
        return topics

    async def get_topic(self, topic_id: int) -> DiscourseTopic | None:
        """Fetch a single topic's metadata."""
        try:
            data = await self._get(f"/t/{topic_id}.json")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        return DiscourseTopic(
            topic_id=data["id"],
            title=data["title"],
            category_id=data.get("category_id"),
            category_name=data.get("category_name"),
            slug=data["slug"],
            posts_count=data["posts_count"],
            created_at=_parse_datetime(data["created_at"]),
            last_posted_at=_parse_datetime(data.get("last_posted_at")),
            tags=data.get("tags", []),
        )

    async def get_post(self, post_id: int) -> DiscoursePost | None:
        """Fetch a single post with raw markdown content."""
        try:
            data = await self._get(f"/posts/{post_id}.json")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        return DiscoursePost(
            post_id=data["id"],
            topic_id=data["topic_id"],
            post_number=data["post_number"],
            username=data["username"],
            raw=data["raw"],  # Original markdown
            reply_to_post_number=data.get("reply_to_post_number"),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )

    async def get_topic_with_posts(
        self, topic_id: int, include_all_posts: bool = True
    ) -> DiscourseTopicWithPosts | None:
        """Fetch a topic with all its posts.

        By default fetches all posts. For large topics, Discourse paginates
        the post_stream - this method handles that.
        """
        try:
            data = await self._get(f"/t/{topic_id}.json")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        topic = DiscourseTopic(
            topic_id=data["id"],
            title=data["title"],
            category_id=data.get("category_id"),
            category_name=data.get("category_name"),
            slug=data["slug"],
            posts_count=data["posts_count"],
            created_at=_parse_datetime(data["created_at"]),
            last_posted_at=_parse_datetime(data.get("last_posted_at")),
            tags=data.get("tags", []),
        )

        # Posts in the initial response
        post_stream = data.get("post_stream", {})
        posts_data = post_stream.get("posts", [])
        all_post_ids = post_stream.get("stream", [])

        posts = []
        loaded_ids = set()

        # Parse posts from initial response
        for p in posts_data:
            posts.append(self._parse_post(p))
            loaded_ids.add(p["id"])

        # Fetch remaining posts if needed
        if include_all_posts:
            remaining_ids = [pid for pid in all_post_ids if pid not in loaded_ids]
            for post_id in remaining_ids:
                post = await self.get_post(post_id)
                if post:
                    posts.append(post)

        # Sort by post_number for consistent ordering
        posts.sort(key=lambda p: p.post_number)

        logger.debug(
            "fetched_topic_with_posts",
            topic_id=topic_id,
            posts_count=len(posts),
        )
        return DiscourseTopicWithPosts(topic=topic, posts=posts)

    def _parse_post(self, data: dict) -> DiscoursePost:
        """Parse a post from inline topic response (has 'cooked' but not 'raw')."""
        # Note: inline posts may not have 'raw', only 'cooked'
        # For these we'll need to fetch individually
        return DiscoursePost(
            post_id=data["id"],
            topic_id=data["topic_id"],
            post_number=data["post_number"],
            username=data["username"],
            raw=data.get("raw", ""),  # May be empty in inline response
            reply_to_post_number=data.get("reply_to_post_number"),
            created_at=_parse_datetime(data["created_at"]),
            updated_at=_parse_datetime(data["updated_at"]),
        )

    async def get_topic_posts_raw(
        self, topic_id: int, post_ids: list[int] | None = None
    ) -> list[DiscoursePost]:
        """Fetch posts with raw markdown content.

        If post_ids is None, fetches all post IDs from the topic.
        """
        # First get the topic to find all post IDs
        data = await self._get(f"/t/{topic_id}.json")
        post_stream = data.get("post_stream", {})

        if post_ids is None:
            post_ids = post_stream.get("stream", [])

        posts = []
        for post_id in post_ids:
            post = await self.get_post(post_id)
            if post:
                posts.append(post)

        posts.sort(key=lambda p: p.post_number)
        return posts

    async def get_categories(self) -> dict[int, str]:
        """Fetch all category mappings."""
        data = await self._get("/categories.json")
        categories = {}
        for cat in data.get("category_list", {}).get("categories", []):
            categories[cat["id"]] = cat["name"]
        return categories


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string from Discourse API."""
    if not value:
        return None
    # Discourse uses ISO format, handle with/without microseconds
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # Fallback for edge cases
        return datetime.fromisoformat(value.rstrip("Z"))
