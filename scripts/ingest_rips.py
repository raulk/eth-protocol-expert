#!/usr/bin/env python3
"""Ingest RIPs (Rollup Improvement Proposals) into the database.

RIPs are optional standards for Ethereum Rollups, stored at ethereum/RIPs.

Usage:
    uv run python scripts/ingest_rips.py
    uv run python scripts/ingest_rips.py --limit 5  # For testing
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.chunking.section_chunker import SectionChunker
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.ingestion.eip_parser import EIPParser
from src.ingestion.rip_loader import RIPLoader
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def ingest_rips(
    batch_size: int = 128,
    limit: int | None = None,
):
    """Ingest RIPs from ethereum/RIPs repo."""
    load_dotenv()

    logger.info("starting_rip_ingestion", batch_size=batch_size, limit=limit)

    # Initialize components
    loader = RIPLoader()
    parser = EIPParser()  # RIPs use similar format to EIPs
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        # Clone/update repo
        logger.info("cloning_or_updating_rips_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        # Load all RIPs
        logger.info("loading_rips")
        loaded_rips = loader.load_all_rips()

        if limit:
            loaded_rips = loaded_rips[:limit]
            logger.info("limiting_to", count=limit)

        # Parse and chunk
        logger.info("parsing_and_chunking_rips", count=len(loaded_rips))
        all_chunks = []
        processed_count = 0

        for rip in loaded_rips:
            try:
                # Create a pseudo-LoadedEIP for the parser
                class LoadedEIPLike:
                    def __init__(self, rip):
                        self.eip_number = rip.rip_number
                        self.raw_content = rip.raw_content
                        self.git_commit = rip.git_commit
                        self.loaded_at = rip.loaded_at
                        self.file_path = rip.file_path

                parsed = parser.parse(LoadedEIPLike(rip))
                chunks = chunker.chunk_eip(parsed)

                # Update document_id in chunks to use rip- prefix
                for chunk in chunks:
                    chunk.document_id = f"rip-{rip.rip_number}"

                all_chunks.extend(chunks)

                # Store document metadata
                await store.store_generic_document(
                    document_id=f"rip-{rip.rip_number}",
                    document_type="rip",
                    title=parsed.title,
                    source="ethereum/RIPs",
                    raw_content=rip.raw_content,
                    author=parsed.author,
                    git_commit=rip.git_commit,
                    metadata={
                        "rip_number": rip.rip_number,
                        "status": parsed.status,
                        "type": parsed.type,
                        "category": parsed.category,
                        "requires": parsed.requires,
                    },
                )
                processed_count += 1

            except Exception as e:
                logger.warning("failed_to_process_rip", rip=rip.rip_number, error=str(e))

        logger.info("chunking_complete", chunks=len(all_chunks), rips=processed_count)

        # Embed and store chunks
        if all_chunks:
            logger.info("embedding_chunks", total=len(all_chunks))
            all_embedded = []

            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                embedded = embedder.embed_chunks(batch)
                all_embedded.extend(embedded)

                if len(all_embedded) >= batch_size * 2:
                    await store.store_embedded_chunks(
                        all_embedded,
                        metadata={"document_type": "rip"},
                    )
                    logger.info(
                        "stored_batch", embedded=i + len(batch), total=len(all_chunks)
                    )
                    all_embedded = []

            # Store remaining
            if all_embedded:
                await store.store_embedded_chunks(
                    all_embedded,
                    metadata={"document_type": "rip"},
                )

            # Rebuild index
            await store.reindex_embeddings()

        # Final stats
        final_chunks = await store.count_chunks()
        logger.info("ingestion_complete", rips=processed_count, total_chunks=final_chunks)

    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest RIPs into the database")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding (default: 128)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of RIPs to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_rips(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
