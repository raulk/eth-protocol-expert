#!/usr/bin/env python3
"""Ingest Portal Network specs into the database.

Portal Network enables lightweight protocol access by resource-constrained
devices through multiple peer-to-peer networks providing data access via JSON-RPC.

Usage:
    uv run python scripts/ingest_portal_specs.py
    uv run python scripts/ingest_portal_specs.py --limit 5  # For testing
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.chunking.section_chunker import SectionChunker
from src.embeddings import create_embedder
from src.ingestion import MarkdownSpec, PortalSpecLoader
from src.ingestion.eip_parser import EIPSection
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


@dataclass
class PseudoParsedEIP:
    """Adapter to make MarkdownSpec compatible with SectionChunker."""

    eip_number: int
    title: str
    status: str
    type: str
    category: str | None
    author: str
    created: str
    requires: list[int]
    raw_content: str
    sections: list[EIPSection]
    git_commit: str
    loaded_at: datetime
    frontmatter: dict = field(default_factory=dict)


def spec_to_pseudo_eip(spec: MarkdownSpec, git_commit: str) -> PseudoParsedEIP:
    """Convert a MarkdownSpec to a pseudo-ParsedEIP for chunking."""
    # Extract sections from the markdown content
    sections = extract_sections(spec.content)

    return PseudoParsedEIP(
        eip_number=0,  # Not used for portal specs
        title=spec.title,
        status="Active",
        type="Specification",
        category=spec.category,
        author="",
        created="",
        requires=[],
        raw_content=spec.content,
        sections=sections,
        git_commit=git_commit,
        loaded_at=datetime.now(),
    )


def extract_sections(content: str) -> list[EIPSection]:
    """Extract sections from markdown content based on headings."""
    import re

    sections = []
    # Find all headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        name = match.group(2).strip()
        start = match.end()

        # End is the start of the next heading or end of content
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(content)

        section_content = content[start:end].strip()
        if section_content:
            sections.append(
                EIPSection(
                    name=name,
                    level=level,
                    content=section_content,
                    start_offset=start,
                    end_offset=end,
                    subsections=[],
                )
            )

    return sections


async def ingest_portal_specs(
    batch_size: int = 128,
    limit: int | None = None,
) -> None:
    """Ingest Portal Network specs from ethereum/portal-network-specs repo."""
    load_dotenv()

    logger.info("starting_portal_spec_ingestion", batch_size=batch_size, limit=limit)

    # Initialize components
    loader = PortalSpecLoader()
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = create_embedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        # Clone/update repo
        logger.info("cloning_or_updating_portal_specs_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        # Load all specs
        logger.info("loading_portal_specs")
        specs = loader.load_all_specs()

        if limit:
            specs = specs[:limit]
            logger.info("limiting_to", count=limit)

        # Process specs
        logger.info("processing_specs", count=len(specs))
        all_chunks = []
        processed_count = 0

        for spec in specs:
            try:
                # Create document_id from spec name
                document_id = f"portal-{spec.name}"

                # Convert to pseudo-ParsedEIP for chunking
                pseudo_eip = spec_to_pseudo_eip(spec, git_commit)
                chunks = chunker.chunk_eip(pseudo_eip)

                # Update document_id in chunks
                for chunk in chunks:
                    chunk.document_id = document_id

                all_chunks.extend(chunks)

                # Store document metadata
                await store.store_generic_document(
                    document_id=document_id,
                    document_type="portal_spec",
                    title=spec.title,
                    source="ethereum/portal-network-specs",
                    raw_content=spec.content,
                    author="",
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
                logger.warning("failed_to_process_spec", spec=spec.name, error=str(e))

        logger.info("chunking_complete", chunks=len(all_chunks), specs=processed_count)

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
                        metadata={"document_type": "portal_spec"},
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
                    metadata={"document_type": "portal_spec"},
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
    parser = argparse.ArgumentParser(description="Ingest Portal Network specs into the database")
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

    asyncio.run(ingest_portal_specs(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
