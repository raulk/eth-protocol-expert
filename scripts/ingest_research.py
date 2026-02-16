#!/usr/bin/env python3
"""Ingest ethereum/research repository into the database.

The research repository contains Python implementations and documentation
of cryptographic research concepts by Vitalik Buterin and others, including
STARKs, binary fields, erasure coding, Verkle tries, and consensus simulations.

Usage:
    uv run python scripts/ingest_research.py
    uv run python scripts/ingest_research.py --limit 10  # For testing
    uv run python scripts/ingest_research.py --markdown-only  # Skip Python files
    uv run python scripts/ingest_research.py --python-only  # Skip markdown files
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
from src.ingestion.eip_parser import EIPSection
from src.ingestion.research_loader import ResearchLoader
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
    """Pseudo-ParsedEIP for ResearchDoc to work with SectionChunker."""

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


async def ingest_research(
    batch_size: int = 128,
    limit: int | None = None,
    markdown_only: bool = False,
    python_only: bool = False,
) -> None:
    """Ingest research docs from ethereum/research repo."""
    load_dotenv()

    logger.info(
        "starting_research_ingestion",
        batch_size=batch_size,
        limit=limit,
        markdown_only=markdown_only,
        python_only=python_only,
    )

    loader = ResearchLoader(include_python=not markdown_only)
    chunker = SectionChunker(max_tokens=512, overlap_tokens=64)
    embedder = create_embedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        logger.info("cloning_or_updating_research_repo")
        git_commit = loader.clone_or_update()
        logger.info("repo_ready", commit=git_commit[:8])

        logger.info("loading_research_docs")
        if markdown_only:
            docs = loader.load_markdown_only()
        elif python_only:
            docs = loader.load_python_only()
        else:
            docs = loader.load_all_docs()

        if limit:
            docs = docs[:limit]
            logger.info("limiting_to", count=limit)

        logger.info("parsing_and_chunking_docs", count=len(docs))
        all_chunks = []
        processed_count = 0

        for doc in docs:
            try:
                sections = extract_sections(doc.content)

                pseudo_eip = PseudoParsedEIP(
                    eip_number=0,
                    title=doc.title,
                    status="Research",
                    type="Informational",
                    category=doc.category,
                    author="Vitalik Buterin et al",
                    created="",
                    requires=[],
                    discussions_to=None,
                    raw_content=doc.content,
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

                document_id = f"research-{doc.category or 'root'}-{doc.name}"
                document_id = document_id.replace("/", "-")
                if len(document_id) > 120:
                    import hashlib
                    hash_suffix = hashlib.md5(document_id.encode()).hexdigest()[:8]
                    document_id = document_id[:100] + "-" + hash_suffix
                for chunk in chunks:
                    chunk.document_id = document_id

                all_chunks.extend(chunks)

                await store.store_generic_document(
                    document_id=document_id,
                    document_type="research",
                    title=doc.title,
                    source="ethereum/research",
                    raw_content=doc.content,
                    author="Vitalik Buterin et al",
                    git_commit=git_commit,
                    metadata={
                        "name": doc.name,
                        "category": doc.category,
                        "doc_type": doc.doc_type,
                        "file_path": str(doc.file_path),
                    },
                )
                processed_count += 1

            except Exception as e:
                logger.warning("failed_to_process_doc", doc=doc.name, error=str(e))

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
                        metadata={"document_type": "research"},
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
                    metadata={"document_type": "research"},
                )

            await store.reindex_embeddings()

        final_chunks = await store.count_chunks()
        logger.info("ingestion_complete", docs=processed_count, total_chunks=final_chunks)

    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ethereum/research into the database")
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
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Only process markdown files, skip Python",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only process Python files, skip markdown",
    )
    args = parser.parse_args()

    if args.markdown_only and args.python_only:
        parser.error("Cannot specify both --markdown-only and --python-only")

    asyncio.run(
        ingest_research(
            batch_size=args.batch_size,
            limit=args.limit,
            markdown_only=args.markdown_only,
            python_only=args.python_only,
        )
    )


if __name__ == "__main__":
    main()
