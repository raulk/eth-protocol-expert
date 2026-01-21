#!/usr/bin/env python3
"""Ingest ERCs into the database.

ERCs (Ethereum Request for Comments) are application-level standards.
They are loaded from both ethereum/EIPs (category: ERC) and ethereum/ERCs repos.

Usage:
    uv run python scripts/ingest_ercs.py
    uv run python scripts/ingest_ercs.py --limit 10  # For testing
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
from src.ingestion.erc_loader import ERCLoader
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def ingest_ercs(
    batch_size: int = 128,
    limit: int | None = None,
):
    """Ingest ERCs from both ethereum/EIPs and ethereum/ERCs repos."""
    load_dotenv()

    logger.info("starting_erc_ingestion", batch_size=batch_size, limit=limit)

    # Initialize components
    loader = ERCLoader()
    parser = EIPParser()  # ERCs use the same format as EIPs
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        # Clone/update repos
        logger.info("cloning_or_updating_repos")
        commits = loader.clone_or_update()
        logger.info("repos_ready", commits=commits)

        # Load all ERCs (deduplicated)
        logger.info("loading_ercs")
        loaded_ercs = loader.load_all_ercs()

        if limit:
            loaded_ercs = loaded_ercs[:limit]
            logger.info("limiting_to", count=limit)

        # Parse and chunk
        logger.info("parsing_and_chunking_ercs", count=len(loaded_ercs))
        all_chunks = []
        processed_count = 0

        for erc in loaded_ercs:
            try:
                # Create a pseudo-LoadedEIP for the parser
                class LoadedEIPLike:
                    def __init__(self, erc):
                        self.eip_number = erc.erc_number
                        self.raw_content = erc.raw_content
                        self.git_commit = erc.git_commit
                        self.loaded_at = erc.loaded_at
                        self.file_path = erc.file_path

                parsed = parser.parse(LoadedEIPLike(erc))
                chunks = chunker.chunk_eip(parsed)
                all_chunks.extend(chunks)

                # Store document metadata
                await store.store_generic_document(
                    document_id=f"erc-{erc.erc_number}",
                    document_type="erc",
                    title=parsed.title,
                    source=f"ethereum/{erc.source_repo}",
                    raw_content=erc.raw_content,
                    author=parsed.author,
                    git_commit=erc.git_commit,
                    metadata={
                        "erc_number": erc.erc_number,
                        "status": parsed.status,
                        "type": parsed.type,
                        "category": parsed.category,
                        "requires": parsed.requires,
                        "source_repo": erc.source_repo,
                    },
                )
                processed_count += 1

                if processed_count % 50 == 0:
                    logger.info("processing_progress", processed=processed_count)

            except Exception as e:
                logger.warning("failed_to_process_erc", erc=erc.erc_number, error=str(e))

        logger.info("chunking_complete", chunks=len(all_chunks), ercs=processed_count)

        # Embed and store chunks
        logger.info("embedding_chunks", total=len(all_chunks))
        all_embedded = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            embedded = embedder.embed_chunks(batch)
            all_embedded.extend(embedded)

            if len(all_embedded) >= batch_size * 2:
                await store.store_embedded_chunks(
                    all_embedded,
                    metadata={"document_type": "erc"},
                )
                logger.info("stored_batch", embedded=i + len(batch), total=len(all_chunks))
                all_embedded = []

        # Store remaining
        if all_embedded:
            await store.store_embedded_chunks(
                all_embedded,
                metadata={"document_type": "erc"},
            )

        # Rebuild index
        await store.reindex_embeddings()

        # Final stats
        final_chunks = await store.count_chunks()
        logger.info("ingestion_complete", ercs=processed_count, total_chunks=final_chunks)

    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest ERCs into the database")
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
        help="Limit number of ERCs to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_ercs(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
