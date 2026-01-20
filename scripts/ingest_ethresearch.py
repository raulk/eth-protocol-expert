#!/usr/bin/env python3
"""Ingest ethresear.ch forum topics into the database.

Usage:
    uv run python scripts/ingest_ethresearch.py [--max-topics 1000] [--max-pages 100]
"""

import argparse
import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import ForumChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import EthresearchLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


async def ingest_ethresearch(max_topics: int = 1000, max_pages: int = 100) -> None:
    """Ingest ethresear.ch topics into the database."""
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()
    await store.initialize_schema()

    loader = EthresearchLoader()
    chunker = ForumChunker(max_tokens=512)
    topics_ingested = 0
    chunks_stored = 0

    try:
        async for topic in loader.iter_topics_with_posts(max_topics=max_topics):
            document_id = f"ethresearch-topic-{topic.topic_id}"

            topic_chunks = chunker.chunk_topic(topic)
            if not topic_chunks:
                continue

            standard_chunks = convert_chunks(topic_chunks, document_id)

            embedded = embedder.embed_chunks(standard_chunks)

            await store.store_generic_document(
                document_id=document_id,
                document_type="forum_topic",
                title=topic.title,
                source="ethresearch",
                raw_content="\n\n".join(p.content for p in topic.posts),
                metadata={
                    "topic_id": topic.topic_id,
                    "category": topic.category,
                    "tags": topic.tags,
                    "posts_count": len(topic.posts),
                },
            )

            await store.store_embedded_chunks(embedded)

            topics_ingested += 1
            chunks_stored += len(embedded)
            logger.info(
                "ingested_topic",
                topic_id=topic.topic_id,
                title=topic.title[:50],
                chunks=len(embedded),
            )

    finally:
        await store.close()

    logger.info(
        "ethresearch_ingestion_complete",
        topics=topics_ingested,
        chunks=chunks_stored,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ethresear.ch topics")
    parser.add_argument("--max-topics", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(ingest_ethresearch(max_topics=args.max_topics, max_pages=args.max_pages))


if __name__ == "__main__":
    main()
