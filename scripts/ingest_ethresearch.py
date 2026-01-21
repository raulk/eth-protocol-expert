#!/usr/bin/env python3
"""Ingest ethresear.ch forum topics from cache into the database.

This script reads topics from the local cache and ingests them into the database.
It does NOT fetch from the API - use sync_ethresearch.py to populate the cache first.

Features:
- Pipelined processing: chunking and embedding run concurrently
- Batched embeddings: reduces API overhead
- Skip existing: avoid re-ingesting already processed topics

Usage:
    uv run python scripts/sync_ethresearch.py --max-topics 500  # First sync
    uv run python scripts/ingest_ethresearch.py                  # Then ingest

    uv run python scripts/ingest_ethresearch.py --skip-existing  # Skip already ingested
    uv run python scripts/ingest_ethresearch.py --batch-size 256 # Larger embedding batches
"""

import argparse
import asyncio
from dataclasses import dataclass

import structlog
from dotenv import load_dotenv

from src.chunking import Chunk, ForumChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion import EthresearchLoader, LoadedForumTopic, RawContentCache
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


@dataclass
class ChunkBatch:
    """A batch of chunks from a single topic."""

    topic: LoadedForumTopic
    chunks: list[Chunk]
    document_id: str


async def get_ingested_topic_ids(store: PgVectorStore) -> set[int]:
    """Get set of topic IDs already in the database."""
    query = """
        SELECT DISTINCT (metadata->>'topic_id')::int as topic_id
        FROM documents
        WHERE source = 'ethresearch'
        AND metadata->>'topic_id' IS NOT NULL
    """
    async with store.pool.acquire() as conn:
        rows = await conn.fetch(query)
    return {row["topic_id"] for row in rows}


async def ingest_ethresearch(
    skip_existing: bool = False,
    batch_size: int = 128,
    queue_size: int = 10,
) -> None:
    """Ingest ethresear.ch topics with pipelined chunking and embedding."""
    cache = RawContentCache()
    loader = EthresearchLoader(cache=cache)

    # Check cache stats
    stats = cache.stats("ethresearch")
    if stats["entry_count"] == 0:
        logger.error(
            "cache_empty",
            message="No topics in cache. Run sync_ethresearch.py first.",
        )
        return

    logger.info(
        "starting_ingestion",
        cached_topics=stats["entry_count"],
        skip_existing=skip_existing,
        batch_size=batch_size,
    )

    # Load all topics from cache
    topics = loader.iter_topics_from_cache()

    # Connect to database
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()
    await store.initialize_schema()

    # Get already ingested topics if skipping
    ingested_ids: set[int] = set()
    if skip_existing:
        ingested_ids = await get_ingested_topic_ids(store)
        logger.info("found_existing_topics", count=len(ingested_ids))

    # Filter topics to process
    topics_to_process = [t for t in topics if t.topic_id not in ingested_ids]
    logger.info(
        "topics_to_process",
        total=len(topics),
        skipped=len(ingested_ids),
        to_process=len(topics_to_process),
    )

    if not topics_to_process:
        logger.info("nothing_to_ingest")
        await store.close()
        return

    # Stats tracking
    stats_lock = asyncio.Lock()
    stats = {"topics_ingested": 0, "chunks_stored": 0, "batches_flushed": 0}

    # Queue for producer/consumer pipeline
    queue: asyncio.Queue[ChunkBatch | None] = asyncio.Queue(maxsize=queue_size)

    async def chunk_producer():
        """Chunk topics and feed the queue."""
        chunker = ForumChunker(max_tokens=512)
        for i, topic in enumerate(topics_to_process):
            topic_chunks = chunker.chunk_topic(topic)
            if topic_chunks:
                document_id = f"ethresearch-topic-{topic.topic_id}"
                standard_chunks = convert_chunks(topic_chunks, document_id)
                await queue.put(
                    ChunkBatch(
                        topic=topic,
                        chunks=standard_chunks,
                        document_id=document_id,
                    )
                )
            if (i + 1) % 50 == 0:
                logger.debug("chunked_topics", count=i + 1)
        await queue.put(None)  # Signal done
        logger.info("chunking_complete", total=len(topics_to_process))

    async def embed_consumer():
        """Consume chunks, batch embed, store."""
        pending_batches: list[ChunkBatch] = []
        pending_chunks: list[Chunk] = []

        async def flush_batch():
            """Embed and store accumulated chunks."""
            nonlocal pending_batches, pending_chunks

            if not pending_chunks:
                return

            # Filter out empty chunks (Voyage API rejects empty strings)
            valid_chunks = [c for c in pending_chunks if c.content and c.content.strip()]
            if not valid_chunks:
                pending_batches = []
                pending_chunks = []
                return

            # Embed all chunks in one call (internally batched by VoyageEmbedder)
            embedded = embedder.embed_chunks(valid_chunks)

            # Store chunks
            await store.store_embedded_chunks(embedded)

            # Store document metadata for each topic
            for batch in pending_batches:
                # First post author is the topic author
                topic_author = batch.topic.posts[0].username if batch.topic.posts else None
                await store.store_generic_document(
                    document_id=batch.document_id,
                    document_type="forum_topic",
                    title=batch.topic.title,
                    author=topic_author,
                    source="ethresearch",
                    raw_content="\n\n".join(p.content for p in batch.topic.posts),
                    metadata={
                        "topic_id": batch.topic.topic_id,
                        "slug": batch.topic.slug,
                        "category": batch.topic.category,
                        "tags": batch.topic.tags,
                        "posts_count": len(batch.topic.posts),
                        "url": f"https://ethresear.ch/t/{batch.topic.slug}/{batch.topic.topic_id}",
                    },
                )

            async with stats_lock:
                stats["topics_ingested"] += len(pending_batches)
                stats["chunks_stored"] += len(valid_chunks)
                stats["batches_flushed"] += 1

            logger.info(
                "flushed_batch",
                topics=len(pending_batches),
                chunks=len(valid_chunks),
                total_topics=stats["topics_ingested"],
            )

            pending_batches = []
            pending_chunks = []

        while True:
            batch = await queue.get()
            if batch is None:
                break  # Producer done

            pending_batches.append(batch)
            pending_chunks.extend(batch.chunks)

            # Flush when we hit batch size
            if len(pending_chunks) >= batch_size:
                await flush_batch()

        # Flush remaining
        await flush_batch()
        logger.info("embedding_complete")

    try:
        # Run producer and consumer concurrently
        await asyncio.gather(
            chunk_producer(),
            embed_consumer(),
        )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()
    finally:
        await store.close()

    logger.info(
        "ingestion_complete",
        topics_ingested=stats["topics_ingested"],
        chunks_stored=stats["chunks_stored"],
        batches_flushed=stats["batches_flushed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ethresear.ch topics from cache into database"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip topics that are already in the database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of chunks to batch for embedding (default: 128)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10,
        help="Size of the chunk queue between producer/consumer (default: 10)",
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_ethresearch(
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            queue_size=args.queue_size,
        )
    )


if __name__ == "__main__":
    main()
