#!/usr/bin/env python3
"""Ingest Ethereum Magicians forum topics into the database.

Usage:
    uv run python scripts/ingest_magicians.py [--max-topics 1000]

This script:
1. Connects to ethereum-magicians.org (Discourse API)
2. Iterates through topics with their posts
3. Chunks each topic using ForumChunker
4. Generates embeddings via Voyage AI
5. Stores everything in PostgreSQL
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.chunking import ForumChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import MagiciansLoader
from src.storage import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def ingest_magicians(
    max_topics: int = 1000,
    batch_size: int = 50,
    use_local_embeddings: bool = False,
) -> None:
    """Ingest Ethereum Magicians topics into the database."""
    load_dotenv()

    logger.info(
        "starting_magicians_ingestion",
        max_topics=max_topics,
        batch_size=batch_size,
    )

    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()
    await store.initialize_schema()

    chunker = ForumChunker(max_tokens=512, overlap_tokens=64)
    loader = MagiciansLoader(rate_limit_delay=1.0)

    topics_ingested = 0
    chunks_stored = 0
    topics_skipped = 0

    try:
        async for topic in loader.iter_topics_with_posts(max_topics=max_topics):
            document_id = f"magicians-topic-{topic.topic_id}"

            if not topic.posts:
                topics_skipped += 1
                logger.debug("skipping_empty_topic", topic_id=topic.topic_id)
                continue

            try:
                topic_chunks = chunker.chunk_topic(topic)
                if not topic_chunks:
                    topics_skipped += 1
                    continue

                standard_chunks = convert_chunks(topic_chunks, document_id)

                embedded = embedder.embed_chunks(standard_chunks)

                await store.store_generic_document(
                    document_id=document_id,
                    document_type="forum_topic",
                    title=topic.title,
                    source="magicians",
                    raw_content="\n\n".join(p.content for p in topic.posts if p.content),
                    metadata={
                        "topic_id": topic.topic_id,
                        "slug": topic.slug,
                        "category": topic.category,
                        "tags": topic.tags or [],
                        "posts_count": len(topic.posts),
                        "created_at": topic.created_at.isoformat() if topic.created_at else None,
                        "last_posted_at": topic.last_posted_at.isoformat() if topic.last_posted_at else None,
                    },
                )

                await store.store_embedded_chunks(embedded)

                topics_ingested += 1
                chunks_stored += len(embedded)

                logger.info(
                    "ingested_topic",
                    topic_id=topic.topic_id,
                    title=topic.title[:60] if topic.title else "Untitled",
                    posts=len(topic.posts),
                    chunks=len(embedded),
                    progress=f"{topics_ingested}/{max_topics}",
                )

            except Exception as e:
                logger.error(
                    "failed_to_ingest_topic",
                    topic_id=topic.topic_id,
                    error=str(e),
                )
                topics_skipped += 1

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        await store.close()

    logger.info(
        "magicians_ingestion_complete",
        topics_ingested=topics_ingested,
        topics_skipped=topics_skipped,
        chunks_stored=chunks_stored,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Ethereum Magicians forum topics"
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=1000,
        help="Maximum number of topics to ingest (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for storing chunks (default: 50)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local BGE embeddings instead of Voyage API",
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_magicians(
            max_topics=args.max_topics,
            batch_size=args.batch_size,
            use_local_embeddings=args.local,
        )
    )


if __name__ == "__main__":
    main()
