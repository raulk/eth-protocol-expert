#!/usr/bin/env python3
"""Ingest devp2p specs into the database.

devp2p specs define Ethereum's peer-to-peer networking protocols
including RLPx transport, node discovery, and protocol capabilities.

Usage:
    uv run python scripts/ingest_devp2p.py
    uv run python scripts/ingest_devp2p.py --limit 5  # For testing
"""

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.chunking.section_chunker import SectionChunker
from src.embeddings import create_embedder
from src.ingestion import DevP2PLoader
from src.ingestion.eip_parser import EIPSection
from src.ingestion.markdown_spec_loader import MarkdownSpec
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


@dataclass
class ParsedSpec:
    """Pseudo-ParsedEIP for markdown specs."""

    eip_number: int
    title: str
    status: str
    type: str
    category: str | None
    author: str
    created: str
    requires: list[int]
    discussions_to: str | None
    raw_content: str
    abstract: str | None
    motivation: str | None
    specification: str | None
    rationale: str | None
    backwards_compatibility: str | None
    security_considerations: str | None
    sections: list[EIPSection]
    git_commit: str
    loaded_at: datetime
    frontmatter: dict = field(default_factory=dict)


def parse_markdown_sections(content: str) -> list[EIPSection]:
    """Parse markdown content into sections based on headings."""
    sections = []
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(heading_pattern.finditer(content))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        name = match.group(2).strip()
        start_offset = match.start()

        if i + 1 < len(matches):
            end_offset = matches[i + 1].start()
        else:
            end_offset = len(content)

        section_content = content[match.end() : end_offset].strip()

        sections.append(
            EIPSection(
                name=name,
                level=level,
                content=section_content,
                start_offset=start_offset,
                end_offset=end_offset,
                subsections=[],
            )
        )

    return sections


def spec_to_parsed_eip(spec: MarkdownSpec, git_commit: str) -> ParsedSpec:
    """Convert a MarkdownSpec to a ParsedEIP-like structure for chunking."""
    sections = parse_markdown_sections(spec.content)

    return ParsedSpec(
        eip_number=0,  # Will be overridden in document_id
        title=spec.title,
        status="Final",  # devp2p specs are generally stable
        type="Networking",
        category=spec.category,
        author="Ethereum Foundation",
        created="",
        requires=[],
        discussions_to=None,
        raw_content=spec.content,
        abstract=None,
        motivation=None,
        specification=None,
        rationale=None,
        backwards_compatibility=None,
        security_considerations=None,
        sections=sections,
        git_commit=git_commit,
        loaded_at=datetime.now(UTC),
    )


async def ingest_devp2p(
    batch_size: int = 128,
    limit: int | None = None,
) -> None:
    """Ingest devp2p specs from ethereum/devp2p repo."""
    load_dotenv()

    logger.info("starting_devp2p_ingestion", batch_size=batch_size, limit=limit)

    loader = DevP2PLoader()
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = create_embedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        logger.info("cloning_or_updating_devp2p_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        logger.info("loading_devp2p_specs")
        specs = loader.load_all_specs()

        if limit:
            specs = specs[:limit]
            logger.info("limiting_to", count=limit)

        logger.info("parsing_and_chunking_specs", count=len(specs))
        all_chunks = []
        processed_count = 0

        for spec in specs:
            try:
                parsed = spec_to_parsed_eip(spec, git_commit)
                chunks = chunker.chunk_eip(parsed)

                # Generate a unique document_id from the spec name
                document_id = f"devp2p-{spec.name}"

                # Update document_id in chunks
                for chunk in chunks:
                    chunk.document_id = document_id

                all_chunks.extend(chunks)

                # Store document metadata
                await store.store_generic_document(
                    document_id=document_id,
                    document_type="devp2p_spec",
                    title=spec.title,
                    source="ethereum/devp2p",
                    raw_content=spec.content,
                    git_commit=git_commit,
                    metadata={
                        "name": spec.name,
                        "category": spec.category,
                        "title": spec.title,
                    },
                )
                processed_count += 1

                if processed_count % 10 == 0:
                    logger.info("processing_progress", processed=processed_count)

            except Exception as e:
                logger.warning(
                    "failed_to_process_spec", spec=spec.name, error=str(e)
                )

        logger.info(
            "chunking_complete", chunks=len(all_chunks), specs=processed_count
        )

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
                        metadata={"document_type": "devp2p_spec"},
                    )
                    logger.info(
                        "stored_batch",
                        embedded=i + len(batch),
                        total=len(all_chunks),
                    )
                    all_embedded = []

            # Store remaining
            if all_embedded:
                await store.store_embedded_chunks(
                    all_embedded,
                    metadata={"document_type": "devp2p_spec"},
                )

            # Rebuild index
            await store.reindex_embeddings()

        # Final stats
        final_chunks = await store.count_chunks()
        logger.info(
            "ingestion_complete",
            specs=processed_count,
            total_chunks=final_chunks,
        )

    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest devp2p specs into the database"
    )
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
        help="Limit number of specs to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_devp2p(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
