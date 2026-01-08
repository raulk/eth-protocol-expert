"""Forum Graph - Build reply relationships from forum posts."""

from dataclasses import dataclass
from typing import Any

import structlog

from ..ingestion.ethresearch_loader import LoadedForumPost, LoadedForumTopic
from .falkordb_store import FalkorDBStore, QueryResult

logger = structlog.get_logger()


@dataclass
class ForumNode:
    """A forum post node in the graph."""
    source: str
    topic_id: int
    post_id: int
    post_number: int
    username: str
    created_at: str


@dataclass
class ReplyEdge:
    """A REPLIES_TO relationship between posts."""
    from_post: int
    to_post: int
    topic_id: int


class ForumGraphBuilder:
    """Build forum post reply graphs in FalkorDB.

    Creates:
    - ForumTopic nodes
    - ForumPost nodes
    - REPLIES_TO edges between posts
    - POSTED_IN edges from posts to topics
    """

    def __init__(self, store: FalkorDBStore):
        self.store = store

    def initialize_forum_schema(self):
        """Create indexes for forum-related nodes."""
        # Index on ForumPost source+post_id for lookups
        try:
            self.store.query(
                "CREATE INDEX FOR (p:ForumPost) ON (p.source, p.post_id)"
            )
            logger.info("created_forum_post_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

        # Index on ForumTopic source+topic_id
        try:
            self.store.query(
                "CREATE INDEX FOR (t:ForumTopic) ON (t.source, t.topic_id)"
            )
            logger.info("created_forum_topic_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

        # Index on username for author lookups
        try:
            self.store.query("CREATE INDEX FOR (p:ForumPost) ON (p.username)")
            logger.info("created_forum_username_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

    def create_topic_node(self, topic: LoadedForumTopic) -> QueryResult:
        """Create a ForumTopic node."""
        cypher = """
            MERGE (t:ForumTopic {source: $source, topic_id: $topic_id})
            ON CREATE SET
                t.title = $title,
                t.slug = $slug,
                t.category = $category,
                t.posts_count = $posts_count,
                t.created_at = $created_at
            ON MATCH SET
                t.title = $title,
                t.posts_count = $posts_count
            RETURN t
        """
        return self.store.query(
            cypher,
            params={
                "source": topic.source,
                "topic_id": topic.topic_id,
                "title": topic.title,
                "slug": topic.slug,
                "category": topic.category,
                "posts_count": topic.posts_count,
                "created_at": topic.created_at.isoformat() if topic.created_at else None,
            },
        )

    def create_post_node(self, post: LoadedForumPost) -> QueryResult:
        """Create a ForumPost node."""
        cypher = """
            MERGE (p:ForumPost {source: $source, post_id: $post_id})
            ON CREATE SET
                p.topic_id = $topic_id,
                p.post_number = $post_number,
                p.username = $username,
                p.reply_to_post_number = $reply_to_post_number,
                p.created_at = $created_at
            ON MATCH SET
                p.post_number = $post_number,
                p.username = $username
            RETURN p
        """
        return self.store.query(
            cypher,
            params={
                "source": post.source,
                "post_id": post.post_id,
                "topic_id": post.topic_id,
                "post_number": post.post_number,
                "username": post.username,
                "reply_to_post_number": post.reply_to_post_number,
                "created_at": post.created_at.isoformat() if post.created_at else None,
            },
        )

    def create_posted_in_relationship(
        self, post: LoadedForumPost
    ) -> QueryResult:
        """Create a POSTED_IN relationship from post to topic."""
        cypher = """
            MATCH (p:ForumPost {source: $source, post_id: $post_id})
            MATCH (t:ForumTopic {source: $source, topic_id: $topic_id})
            MERGE (p)-[r:POSTED_IN]->(t)
            RETURN p, r, t
        """
        return self.store.query(
            cypher,
            params={
                "source": post.source,
                "post_id": post.post_id,
                "topic_id": post.topic_id,
            },
        )

    def create_replies_to_relationship(
        self,
        source: str,
        topic_id: int,
        from_post_number: int,
        to_post_number: int,
    ) -> QueryResult:
        """Create a REPLIES_TO relationship between posts in the same topic."""
        cypher = """
            MATCH (from:ForumPost {source: $source, topic_id: $topic_id, post_number: $from_post_number})
            MATCH (to:ForumPost {source: $source, topic_id: $topic_id, post_number: $to_post_number})
            MERGE (from)-[r:REPLIES_TO]->(to)
            RETURN from, r, to
        """
        return self.store.query(
            cypher,
            params={
                "source": source,
                "topic_id": topic_id,
                "from_post_number": from_post_number,
                "to_post_number": to_post_number,
            },
        )

    def build_topic_graph(self, topic: LoadedForumTopic) -> dict[str, int]:
        """Build complete graph for a topic including all reply relationships.

        Returns counts of nodes and relationships created.
        """
        stats = {
            "topics_created": 0,
            "posts_created": 0,
            "posted_in_created": 0,
            "replies_to_created": 0,
        }

        # Create topic node
        result = self.create_topic_node(topic)
        stats["topics_created"] += result.nodes_created

        # Create post nodes and relationships
        for post in topic.posts:
            # Create post node
            result = self.create_post_node(post)
            stats["posts_created"] += result.nodes_created

            # Create POSTED_IN relationship
            result = self.create_posted_in_relationship(post)
            stats["posted_in_created"] += result.relationships_created

            # Create REPLIES_TO relationship if this is a reply
            if post.reply_to_post_number is not None:
                result = self.create_replies_to_relationship(
                    source=post.source,
                    topic_id=post.topic_id,
                    from_post_number=post.post_number,
                    to_post_number=post.reply_to_post_number,
                )
                stats["replies_to_created"] += result.relationships_created

        logger.info(
            "built_topic_graph",
            source=topic.source,
            topic_id=topic.topic_id,
            **stats,
        )
        return stats

    def get_post_replies(
        self, source: str, topic_id: int, post_number: int
    ) -> list[dict[str, Any]]:
        """Get all direct replies to a specific post."""
        result = self.store.query(
            """
            MATCH (reply:ForumPost)-[:REPLIES_TO]->(post:ForumPost {
                source: $source, topic_id: $topic_id, post_number: $post_number
            })
            RETURN reply.post_number, reply.username, reply.created_at
            ORDER BY reply.post_number
            """,
            params={
                "source": source,
                "topic_id": topic_id,
                "post_number": post_number,
            },
        )
        return [
            {
                "post_number": row[0],
                "username": row[1],
                "created_at": row[2],
            }
            for row in result.result_set
        ]

    def get_reply_thread(
        self, source: str, topic_id: int, post_number: int, max_depth: int = 10
    ) -> list[dict[str, Any]]:
        """Get the reply chain leading up to a post (parents)."""
        result = self.store.query(
            f"""
            MATCH path = (post:ForumPost {{
                source: $source, topic_id: $topic_id, post_number: $post_number
            }})-[:REPLIES_TO*1..{max_depth}]->(parent:ForumPost)
            RETURN parent.post_number, parent.username, parent.created_at,
                   length(path) as depth
            ORDER BY depth
            """,
            params={
                "source": source,
                "topic_id": topic_id,
                "post_number": post_number,
            },
        )
        return [
            {
                "post_number": row[0],
                "username": row[1],
                "created_at": row[2],
                "depth": row[3],
            }
            for row in result.result_set
        ]

    def get_user_posts(
        self, username: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get all posts by a specific username."""
        result = self.store.query(
            """
            MATCH (p:ForumPost {username: $username})-[:POSTED_IN]->(t:ForumTopic)
            RETURN p.source, p.topic_id, p.post_number, t.title, p.created_at
            ORDER BY p.created_at DESC
            LIMIT $limit
            """,
            params={"username": username, "limit": limit},
        )
        return [
            {
                "source": row[0],
                "topic_id": row[1],
                "post_number": row[2],
                "topic_title": row[3],
                "created_at": row[4],
            }
            for row in result.result_set
        ]

    def get_topic_participants(
        self, source: str, topic_id: int
    ) -> list[dict[str, Any]]:
        """Get all participants in a topic with post counts."""
        result = self.store.query(
            """
            MATCH (p:ForumPost {source: $source, topic_id: $topic_id})
            RETURN p.username, count(p) as post_count
            ORDER BY post_count DESC
            """,
            params={"source": source, "topic_id": topic_id},
        )
        return [
            {"username": row[0], "post_count": row[1]}
            for row in result.result_set
        ]

    def count_forum_nodes(self) -> dict[str, int]:
        """Count forum-related nodes."""
        counts = {}

        result = self.store.query("MATCH (t:ForumTopic) RETURN count(t)")
        counts["topics"] = result.result_set[0][0] if result.result_set else 0

        result = self.store.query("MATCH (p:ForumPost) RETURN count(p)")
        counts["posts"] = result.result_set[0][0] if result.result_set else 0

        return counts

    def count_forum_relationships(self) -> dict[str, int]:
        """Count forum-related relationships."""
        counts = {}

        result = self.store.query("MATCH ()-[r:POSTED_IN]->() RETURN count(r)")
        counts["posted_in"] = result.result_set[0][0] if result.result_set else 0

        result = self.store.query("MATCH ()-[r:REPLIES_TO]->() RETURN count(r)")
        counts["replies_to"] = result.result_set[0][0] if result.result_set else 0

        return counts
