#!/usr/bin/env python3
"""Ingest EIPs into the database.

Usage:
    python scripts/ingest_eips.py [--data-dir DATA_DIR] [--use-section-chunking]

This script:
1. Clones/updates the ethereum/EIPs repository
2. Parses all EIP markdown files
3. Chunks them (fixed or section-aware)
4. Generates embeddings
5. Stores everything in PostgreSQL
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.chunking.fixed_chunker import FixedChunker
from src.chunking.section_chunker import SectionChunker
from src.embeddings import create_embedder
from src.graph import EIPGraphBuilder, FalkorDBStore
from src.ingestion.eip_loader import EIPLoader
from src.ingestion.eip_parser import EIPParser
from src.storage.pg_vector_store import PgVectorStore

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def ingest_eips(
    data_dir: str = "data/eips",
    use_section_chunking: bool = True,
    batch_size: int = 50,
    limit: int | None = None,
):
    """Main ingestion pipeline."""

    logger.info("starting_ingestion", data_dir=data_dir, section_chunking=use_section_chunking)

    # Initialize components
    loader = EIPLoader(data_dir=data_dir)
    parser = EIPParser()

    if use_section_chunking:
        chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    else:
        chunker = FixedChunker(max_tokens=512, overlap_tokens=64)

    embedder = create_embedder()
    store = PgVectorStore()

    # Connect to database
    await store.connect()
    await store.initialize_schema()

    try:
        # Step 1: Clone/update EIPs repo
        logger.info("cloning_or_updating_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        # Step 2: Load all EIPs
        logger.info("loading_eips")
        loaded_eips = loader.load_all_eips()
        logger.info("loaded_eips", count=len(loaded_eips))

        # Apply limit if specified
        if limit:
            loaded_eips = loaded_eips[:limit]
            logger.info("limiting_to", count=limit)

        # Step 3: Parse and chunk all EIPs (CPU-bound, fast)
        logger.info("parsing_and_chunking_all_eips")
        all_chunks = []
        parsed_eips = []

        for i, loaded_eip in enumerate(loaded_eips):
            try:
                parsed = parser.parse(loaded_eip)
                parsed_eips.append(parsed)
                chunks = chunker.chunk_eip(parsed)
                all_chunks.extend(chunks)

                if (i + 1) % 100 == 0:
                    logger.info("chunking_progress", processed=i + 1, total=len(loaded_eips))
            except Exception as e:
                logger.error("failed_to_parse_eip", eip=loaded_eip.eip_number, error=str(e))

        logger.info("chunking_complete", total_chunks=len(all_chunks), total_eips=len(parsed_eips))

        # Step 4: Build EIP graph from relationship data
        logger.info("building_eip_graph")
        graph_store = FalkorDBStore(
            host=os.environ.get("FALKORDB_HOST", "localhost"),
            port=int(os.environ.get("FALKORDB_PORT", "6379")),
        )
        try:
            graph_store.connect()
            graph_store.initialize_schema()

            builder = EIPGraphBuilder(graph_store)
            graph_result = builder.build_from_eips(parsed_eips)
            logger.info(
                "graph_built",
                nodes=graph_result.nodes_created,
                relationships=graph_result.relationships_created,
                requires=graph_result.requires_count,
                supersedes=graph_result.supersedes_count,
            )
        except Exception as e:
            logger.warning("graph_build_failed", error=str(e))
        finally:
            graph_store.close()

        # Step 5: Store all document metadata
        logger.info("storing_documents")
        for parsed in parsed_eips:
            await store.store_document(
                document_id=f"eip-{parsed.eip_number}",
                eip_number=parsed.eip_number,
                title=parsed.title,
                status=parsed.status,
                type_=parsed.type,
                category=parsed.category,
                author=parsed.author,
                created_date=parsed.created,
                requires=parsed.requires,
                raw_content=parsed.raw_content,
                git_commit=git_commit,
            )

        # Step 5: Embed all chunks in large batches (API-bound, main bottleneck)
        logger.info("embedding_all_chunks", total=len(all_chunks))
        embed_batch_size = 128  # Larger batches = fewer API calls
        all_embedded_chunks = []

        for i in range(0, len(all_chunks), embed_batch_size):
            batch = all_chunks[i:i + embed_batch_size]
            embedded = embedder.embed_chunks(batch)
            all_embedded_chunks.extend(embedded)

            # Store periodically to avoid memory issues
            if len(all_embedded_chunks) >= batch_size * 4:
                await store.store_embedded_chunks(all_embedded_chunks, git_commit)
                all_embedded_chunks = []
                logger.info("embedding_progress", embedded=i + len(batch), total=len(all_chunks))

        # Store remaining chunks
        if all_embedded_chunks:
            await store.store_embedded_chunks(all_embedded_chunks, git_commit)

        # Final stats
        final_chunks = await store.count_chunks()
        final_docs = await store.count_documents()

        logger.info(
            "ingestion_complete",
            documents=final_docs,
            chunks=final_chunks,
            git_commit=git_commit[:8],
        )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest EIPs into the database")
    parser.add_argument(
        "--data-dir",
        default="data/eips",
        help="Directory to clone EIPs repo into",
    )
    parser.add_argument(
        "--use-section-chunking",
        action="store_true",
        default=True,
        help="Use section-aware chunking (Phase 1)",
    )
    parser.add_argument(
        "--use-fixed-chunking",
        action="store_true",
        help="Use fixed-size chunking (Phase 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for storing chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of EIPs to process (for testing)",
    )

    args = parser.parse_args()

    use_section = not args.use_fixed_chunking

    asyncio.run(ingest_eips(
        data_dir=args.data_dir,
        use_section_chunking=use_section,
        batch_size=args.batch_size,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
