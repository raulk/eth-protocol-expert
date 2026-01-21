#!/usr/bin/env python3
"""Ingest beacon-apis into the database.

The beacon-apis repository defines the HTTP API for beacon chain nodes,
enabling clients to query chain state, submit blocks, and manage validators.

Usage:
    uv run python scripts/ingest_beacon_apis.py
    uv run python scripts/ingest_beacon_apis.py --limit 5  # For testing
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
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.ingestion.beacon_apis_loader import BeaconAPIsLoader
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
    """Pseudo-ParsedEIP for API endpoint to work with SectionChunker."""

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


def extract_sections(content: str) -> list[EIPSection]:
    """Extract markdown sections from content."""
    sections = []
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(header_pattern.finditer(content))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        name = match.group(2).strip()
        start_offset = match.end() + 1

        if i + 1 < len(matches):
            end_offset = matches[i + 1].start()
        else:
            end_offset = len(content)

        section_content = content[start_offset:end_offset].strip()

        sections.append(
            EIPSection(
                name=name,
                level=level,
                content=section_content,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    return sections


def make_document_id(method: str, path: str) -> str:
    """Create document ID from method and path.

    Format: beacon-api-{method}-{path}
    Example: beacon-api-GET-eth-v1-beacon-genesis
    """
    safe_path = path.strip("/").replace("/", "-").replace("{", "").replace("}", "")
    return f"beacon-api-{method}-{safe_path}"


async def ingest_beacon_apis(
    batch_size: int = 128,
    limit: int | None = None,
) -> None:
    """Ingest beacon-apis from ethereum/beacon-APIs repo."""
    load_dotenv()

    logger.info("starting_beacon_apis_ingestion", batch_size=batch_size, limit=limit)

    loader = BeaconAPIsLoader()
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        logger.info("cloning_or_updating_beacon_apis_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        logger.info("loading_beacon_apis_spec")
        specs = loader.load_specs()

        if not specs:
            logger.error("no_specs_found")
            return

        spec = specs[0]
        endpoints = spec.endpoints

        if limit:
            endpoints = endpoints[:limit]
            logger.info("limiting_to", count=limit)

        logger.info("processing_endpoints", count=len(endpoints))
        all_chunks = []
        processed_count = 0

        for endpoint in endpoints:
            try:
                markdown_content = loader.endpoint_to_markdown(endpoint)
                sections = extract_sections(markdown_content)

                title = f"{endpoint.method} {endpoint.path}"
                if endpoint.summary:
                    title = f"{title} - {endpoint.summary}"

                pseudo_eip = PseudoParsedEIP(
                    eip_number=0,
                    title=title,
                    status="Living",
                    type="Standards Track",
                    category=endpoint.tags[0] if endpoint.tags else "Beacon",
                    author="",
                    created="",
                    requires=[],
                    discussions_to=None,
                    raw_content=markdown_content,
                    abstract=endpoint.summary,
                    motivation=None,
                    specification=None,
                    rationale=None,
                    backwards_compatibility=None,
                    security_considerations=None,
                    sections=sections,
                    git_commit=git_commit,
                    loaded_at=datetime.now(UTC),
                )

                chunks = chunker.chunk_eip(pseudo_eip)

                document_id = make_document_id(endpoint.method, endpoint.path)
                for chunk in chunks:
                    chunk.document_id = document_id

                all_chunks.extend(chunks)

                await store.store_generic_document(
                    document_id=document_id,
                    document_type="beacon_api",
                    title=title,
                    source="ethereum/beacon-apis",
                    raw_content=markdown_content,
                    author=None,
                    git_commit=git_commit,
                    metadata={
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "tags": endpoint.tags,
                        "operation_id": endpoint.operation_id,
                        "summary": endpoint.summary,
                    },
                )
                processed_count += 1

            except Exception as e:
                logger.warning(
                    "failed_to_process_endpoint",
                    endpoint=f"{endpoint.method} {endpoint.path}",
                    error=str(e),
                )

        logger.info("chunking_complete", chunks=len(all_chunks), endpoints=processed_count)

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
                        metadata={"document_type": "beacon_api"},
                    )
                    logger.info(
                        "stored_batch",
                        embedded=i + len(batch),
                        total=len(all_chunks),
                    )
                    all_embedded = []

            if all_embedded:
                await store.store_embedded_chunks(
                    all_embedded,
                    metadata={"document_type": "beacon_api"},
                )

            await store.reindex_embeddings()

        final_chunks = await store.count_chunks()
        logger.info("ingestion_complete", endpoints=processed_count, total_chunks=final_chunks)

    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest beacon-apis into the database")
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
        help="Limit number of endpoints to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_beacon_apis(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
