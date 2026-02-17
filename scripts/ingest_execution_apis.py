#!/usr/bin/env python3
"""Ingest execution-apis into the database.

Execution-apis defines the JSON-RPC API specification for Ethereum execution
layer clients. This includes eth_* methods (user-facing), engine_* methods
(consensus-execution client interface), and debug_* methods.

Usage:
    uv run python scripts/ingest_execution_apis.py
    uv run python scripts/ingest_execution_apis.py --limit 5  # For testing
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
load_dotenv()  # Must run before any src.* imports

from src.chunking.section_chunker import SectionChunker
from src.embeddings import create_embedder
from src.ingestion.eip_parser import EIPSection
from src.ingestion.execution_apis_loader import ExecutionAPIsLoader
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
    """Pseudo-ParsedEIP for content to work with SectionChunker."""

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


async def ingest_execution_apis(
    batch_size: int = 128,
    limit: int | None = None,
) -> None:
    """Ingest execution-apis from ethereum/execution-apis repo."""

    logger.info("starting_execution_apis_ingestion", batch_size=batch_size, limit=limit)

    loader = ExecutionAPIsLoader()
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = create_embedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        logger.info("cloning_or_updating_execution_apis_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        logger.info("loading_execution_apis")
        all_docs = loader.get_all_markdown()

        if limit:
            all_docs = all_docs[:limit]
            logger.info("limiting_to", count=limit)

        logger.info("parsing_and_chunking_docs", count=len(all_docs))
        all_chunks = []
        processed_count = 0

        for doc_type, title, content, source_path in all_docs:
            try:
                sections = extract_sections(content)

                pseudo_eip = PseudoParsedEIP(
                    eip_number=0,
                    title=title,
                    status="Living",
                    type="Standards Track",
                    category=doc_type,
                    author="",
                    created="",
                    requires=[],
                    discussions_to=None,
                    raw_content=content,
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

                chunks = chunker.chunk_eip(pseudo_eip)

                safe_title = re.sub(r"[^a-zA-Z0-9_-]", "-", title.lower())
                document_id = f"exec-api-{doc_type}-{safe_title}"
                for chunk in chunks:
                    chunk.document_id = document_id

                all_chunks.extend(chunks)

                await store.store_generic_document(
                    document_id=document_id,
                    document_type="execution_api",
                    title=title,
                    source="ethereum/execution-apis",
                    raw_content=content,
                    author=None,
                    git_commit=git_commit,
                    metadata={
                        "doc_type": doc_type,
                        "source_file": str(source_path),
                    },
                )
                processed_count += 1

            except Exception as e:
                logger.warning("failed_to_process_doc", title=title, error=str(e))

        logger.info("chunking_complete", chunks=len(all_chunks), docs=processed_count)

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
                        metadata={"document_type": "execution_api"},
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
                    metadata={"document_type": "execution_api"},
                )

            await store.reindex_embeddings()

        final_chunks = await store.count_chunks()
        logger.info("ingestion_complete", docs=processed_count, total_chunks=final_chunks)

    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest execution-apis into the database")
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
        help="Limit number of docs to process (for testing)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_execution_apis(batch_size=args.batch_size, limit=args.limit))


if __name__ == "__main__":
    main()
