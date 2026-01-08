"""Cross-Reference - Link forum discussions to EIPs they mention."""

import re
from dataclasses import dataclass
from typing import Any

import structlog

from ..ingestion.ethresearch_loader import LoadedForumPost, LoadedForumTopic
from .falkordb_store import FalkorDBStore, QueryResult

logger = structlog.get_logger()

# Pattern to match EIP references like "EIP-4844", "EIP 4844", "EIP4844"
EIP_PATTERN = re.compile(r"\bEIP[- ]?(\d+)\b", re.IGNORECASE)


@dataclass
class EIPMention:
    """A mention of an EIP in forum content."""
    eip_number: int
    source: str
    topic_id: int
    post_id: int
    post_number: int
    context: str  # Surrounding text for the mention


class CrossReferenceBuilder:
    """Build cross-references between forum posts and EIPs.

    Creates DISCUSSES edges from ForumPost/ForumTopic nodes to EIP nodes
    when EIP-XXXX patterns are detected in the content.
    """

    def __init__(self, store: FalkorDBStore):
        self.store = store

    def extract_eip_mentions(self, text: str) -> list[int]:
        """Extract all EIP numbers mentioned in text."""
        matches = EIP_PATTERN.findall(text)
        return sorted(set(int(m) for m in matches))

    def extract_eip_mentions_with_context(
        self, text: str, context_chars: int = 100
    ) -> list[tuple[int, str]]:
        """Extract EIP mentions with surrounding context."""
        results = []
        for match in EIP_PATTERN.finditer(text):
            eip_number = int(match.group(1))
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            context = text[start:end]
            # Clean up context
            context = " ".join(context.split())
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            results.append((eip_number, context))
        return results

    def find_post_eip_mentions(self, post: LoadedForumPost) -> list[EIPMention]:
        """Find all EIP mentions in a post."""
        mentions = []
        for eip_number, context in self.extract_eip_mentions_with_context(post.content):
            mentions.append(EIPMention(
                eip_number=eip_number,
                source=post.source,
                topic_id=post.topic_id,
                post_id=post.post_id,
                post_number=post.post_number,
                context=context,
            ))
        return mentions

    def find_topic_eip_mentions(self, topic: LoadedForumTopic) -> list[int]:
        """Find all unique EIPs mentioned anywhere in a topic (title + all posts)."""
        mentioned_eips = set()

        # Check title
        mentioned_eips.update(self.extract_eip_mentions(topic.title))

        # Check all posts
        for post in topic.posts:
            mentioned_eips.update(self.extract_eip_mentions(post.content))

        return sorted(mentioned_eips)

    def create_post_discusses_eip(
        self, post: LoadedForumPost, eip_number: int
    ) -> QueryResult:
        """Create a DISCUSSES relationship from a post to an EIP."""
        cypher = """
            MATCH (p:ForumPost {source: $source, post_id: $post_id})
            MATCH (e:EIP {number: $eip_number})
            MERGE (p)-[r:DISCUSSES]->(e)
            RETURN p, r, e
        """
        return self.store.query(
            cypher,
            params={
                "source": post.source,
                "post_id": post.post_id,
                "eip_number": eip_number,
            },
        )

    def create_topic_discusses_eip(
        self, topic: LoadedForumTopic, eip_number: int
    ) -> QueryResult:
        """Create a DISCUSSES relationship from a topic to an EIP."""
        cypher = """
            MATCH (t:ForumTopic {source: $source, topic_id: $topic_id})
            MATCH (e:EIP {number: $eip_number})
            MERGE (t)-[r:DISCUSSES]->(e)
            RETURN t, r, e
        """
        return self.store.query(
            cypher,
            params={
                "source": topic.source,
                "topic_id": topic.topic_id,
                "eip_number": eip_number,
            },
        )

    def build_topic_cross_references(
        self, topic: LoadedForumTopic
    ) -> dict[str, Any]:
        """Build all cross-references for a topic.

        Creates DISCUSSES edges at both topic and post level.
        Returns stats about references created.
        """
        stats = {
            "eips_found": [],
            "topic_discusses_created": 0,
            "post_discusses_created": 0,
        }

        # Find all EIPs mentioned in the topic
        topic_eips = self.find_topic_eip_mentions(topic)
        stats["eips_found"] = topic_eips

        # Create topic-level DISCUSSES edges
        for eip_number in topic_eips:
            result = self.create_topic_discusses_eip(topic, eip_number)
            stats["topic_discusses_created"] += result.relationships_created

        # Create post-level DISCUSSES edges
        for post in topic.posts:
            post_eips = self.extract_eip_mentions(post.content)
            for eip_number in post_eips:
                result = self.create_post_discusses_eip(post, eip_number)
                stats["post_discusses_created"] += result.relationships_created

        if topic_eips:
            logger.info(
                "built_cross_references",
                source=topic.source,
                topic_id=topic.topic_id,
                eips=topic_eips,
                topic_edges=stats["topic_discusses_created"],
                post_edges=stats["post_discusses_created"],
            )

        return stats

    def get_eip_discussions(
        self, eip_number: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get all forum topics that discuss a specific EIP."""
        result = self.store.query(
            """
            MATCH (t:ForumTopic)-[:DISCUSSES]->(e:EIP {number: $eip_number})
            RETURN t.source, t.topic_id, t.title, t.posts_count, t.created_at
            ORDER BY t.posts_count DESC
            LIMIT $limit
            """,
            params={"eip_number": eip_number, "limit": limit},
        )
        return [
            {
                "source": row[0],
                "topic_id": row[1],
                "title": row[2],
                "posts_count": row[3],
                "created_at": row[4],
            }
            for row in result.result_set
        ]

    def get_eip_post_mentions(
        self, eip_number: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get all posts that mention a specific EIP."""
        result = self.store.query(
            """
            MATCH (p:ForumPost)-[:DISCUSSES]->(e:EIP {number: $eip_number})
            MATCH (p)-[:POSTED_IN]->(t:ForumTopic)
            RETURN p.source, p.topic_id, p.post_number, p.username, p.created_at, t.title
            ORDER BY p.created_at DESC
            LIMIT $limit
            """,
            params={"eip_number": eip_number, "limit": limit},
        )
        return [
            {
                "source": row[0],
                "topic_id": row[1],
                "post_number": row[2],
                "username": row[3],
                "created_at": row[4],
                "topic_title": row[5],
            }
            for row in result.result_set
        ]

    def get_related_eips_by_discussion(
        self, eip_number: int
    ) -> list[dict[str, Any]]:
        """Find EIPs that are frequently discussed together in the same topics.

        This can reveal relationships not captured in the EIP frontmatter.
        """
        result = self.store.query(
            """
            MATCH (t:ForumTopic)-[:DISCUSSES]->(e1:EIP {number: $eip_number})
            MATCH (t)-[:DISCUSSES]->(e2:EIP)
            WHERE e2.number <> $eip_number
            RETURN e2.number, e2.title, count(t) as co_mentions
            ORDER BY co_mentions DESC
            LIMIT 20
            """,
            params={"eip_number": eip_number},
        )
        return [
            {
                "eip_number": row[0],
                "title": row[1],
                "co_mentions": row[2],
            }
            for row in result.result_set
        ]

    def get_most_discussed_eips(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get EIPs with the most forum discussions."""
        result = self.store.query(
            """
            MATCH (t:ForumTopic)-[:DISCUSSES]->(e:EIP)
            RETURN e.number, e.title, e.status, count(t) as discussion_count
            ORDER BY discussion_count DESC
            LIMIT $limit
            """,
            params={"limit": limit},
        )
        return [
            {
                "eip_number": row[0],
                "title": row[1],
                "status": row[2],
                "discussion_count": row[3],
            }
            for row in result.result_set
        ]

    def count_discusses_relationships(self) -> int:
        """Count total DISCUSSES relationships."""
        result = self.store.query("MATCH ()-[r:DISCUSSES]->() RETURN count(r)")
        return result.result_set[0][0] if result.result_set else 0
