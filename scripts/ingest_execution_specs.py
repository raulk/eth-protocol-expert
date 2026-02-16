#!/usr/bin/env python3
"""Ingest ethereum/execution-specs repository.

Usage:
    uv run python scripts/ingest_execution_specs.py
"""

import asyncio
import hashlib

import structlog
import tiktoken
from dotenv import load_dotenv

from src.chunking import Chunk
from src.embeddings import create_embedder
from src.ingestion import ExecutionSpecLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


def chunk_python_code(
    content: str,
    document_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk Python code into fixed-size pieces with overlap.

    Args:
        content: Python source code content.
        document_id: Document identifier for the spec.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(content)

    if len(tokens) <= max_tokens:
        return [
            Chunk(
                chunk_id=hashlib.sha256(content.encode()).hexdigest()[:16],
                document_id=document_id,
                content=content,
                token_count=len(tokens),
                chunk_index=0,
                section_path=None,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(
            Chunk(
                chunk_id=hashlib.sha256(
                    f"{document_id}-{chunk_index}-{chunk_text}".encode()
                ).hexdigest()[:16],
                document_id=document_id,
                content=chunk_text,
                token_count=len(chunk_tokens),
                chunk_index=chunk_index,
                section_path=None,
            )
        )

        chunk_index += 1
        start = end - overlap_tokens

        if start >= len(tokens) - overlap_tokens:
            break

    return chunks


async def ingest_execution_specs() -> None:
    """Ingest execution specs into the database."""
    loader = ExecutionSpecLoader()
    embedder = create_embedder()
    store = PgVectorStore()
    await store.connect()

    specs_ingested = 0
    chunks_stored = 0

    try:
        git_commit = loader.clone_or_update()
        specs = loader.load_all_specs()
        logger.info("found_specs", count=len(specs), git_commit=git_commit[:8])

        for spec in specs:
            document_id = f"execution-spec-{spec.fork}-{spec.module_path.replace('.', '-')}"

            chunks = chunk_python_code(
                content=spec.content,
                document_id=document_id,
                max_tokens=512,
                overlap_tokens=64,
            )
            if not chunks:
                continue

            embedded = embedder.embed_chunks(chunks)

            await store.store_generic_document(
                document_id=document_id,
                document_type="execution_spec",
                title=f"{spec.fork}: {spec.module_path}",
                source="ethereum/execution-specs",
                raw_content=spec.content,
                git_commit=git_commit,
                metadata={
                    "fork": spec.fork,
                    "module_path": spec.module_path,
                    "file_path": str(spec.file_path),
                },
            )

            await store.store_embedded_chunks(embedded, git_commit=git_commit)

            specs_ingested += 1
            chunks_stored += len(embedded)
            logger.info(
                "ingested_spec",
                fork=spec.fork,
                module_path=spec.module_path,
                chunks=len(embedded),
            )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        await store.close()

    logger.info(
        "execution_specs_ingestion_complete",
        specs=specs_ingested,
        chunks=chunks_stored,
    )


def main() -> None:
    asyncio.run(ingest_execution_specs())


if __name__ == "__main__":
    main()
