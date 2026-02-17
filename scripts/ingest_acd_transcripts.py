#!/usr/bin/env python3
"""Ingest AllCoreDevs call transcripts.

Usage:
    uv run python scripts/ingest_acd_transcripts.py [--limit N] [--local]

This script:
1. Clones/updates the ethereum/pm repository
2. Lists all ACD transcripts
3. Chunks them with speaker-aware chunking
4. Generates embeddings
5. Stores everything in PostgreSQL
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.chunking import TranscriptChunker, convert_chunks
from src.embeddings import create_embedder
from src.ingestion import ACDTranscriptLoader
from src.storage import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def ingest_acd_transcripts(
    data_dir: str = "data/pm",
    limit: int | None = None,
) -> None:
    """Ingest ACD transcripts into the database."""

    logger.info(
        "starting_acd_ingestion",
        data_dir=data_dir,
        limit=limit,
    )

    loader = ACDTranscriptLoader(repo_path=data_dir)
    chunker = TranscriptChunker(max_tokens=512)

    embedder = create_embedder()

    store = PgVectorStore()
    await store.connect()

    try:
        if not loader.clone_repo():
            logger.error("failed_to_clone_pm_repo")
            return

        transcripts = loader.list_transcripts()
        logger.info("found_transcripts", count=len(transcripts))

        if limit:
            transcripts = transcripts[:limit]
            logger.info("limiting_to", count=limit)

        ingested_count = 0
        total_chunks = 0

        for transcript in transcripts:
            document_id = f"acd-call-{transcript.call_number}"

            transcript_chunks = chunker.chunk(transcript)
            if not transcript_chunks:
                logger.warning(
                    "no_chunks_from_transcript",
                    call_number=transcript.call_number,
                )
                continue

            standard_chunks = convert_chunks(transcript_chunks, document_id)

            embedded = embedder.embed_chunks(standard_chunks)

            await store.store_generic_document(
                document_id=document_id,
                document_type="acd_transcript",
                title=transcript.title or f"ACD Call #{transcript.call_number}",
                source="ethereum/pm",
                raw_content=transcript.raw_markdown,
                metadata={
                    "call_number": transcript.call_number,
                    "date": transcript.date.isoformat() if transcript.date else None,
                    "speakers": transcript.speakers,
                },
            )

            await store.store_embedded_chunks(embedded)

            ingested_count += 1
            total_chunks += len(embedded)

            logger.info(
                "ingested_transcript",
                call_number=transcript.call_number,
                chunks=len(embedded),
            )

        logger.info(
            "acd_ingestion_complete",
            transcripts=ingested_count,
            total_chunks=total_chunks,
        )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ACD transcripts into the database")
    parser.add_argument(
        "--data-dir",
        default="data/pm",
        help="Directory to clone ethereum/pm repo into",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of transcripts to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_acd_transcripts(
            data_dir=args.data_dir,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
