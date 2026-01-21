#!/usr/bin/env python3
"""Ingest ethereum/consensus-specs repository.

Usage:
    uv run python scripts/ingest_consensus_specs.py
"""

import asyncio
import hashlib
import re

import structlog
from dotenv import load_dotenv

from src.chunking import Chunk
from src.embeddings import VoyageEmbedder
from src.ingestion import ConsensusSpecLoader
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


def chunk_markdown(
    content: str,
    document_id: str,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Chunk markdown content into standard Chunk objects.

    Splits on section boundaries (headings) and paragraph breaks.
    Keeps code blocks atomic when possible.
    """
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    def hash_content(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    chunks: list[Chunk] = []
    chunk_index = 0

    # Split by major sections (## headings)
    section_pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    parts = section_pattern.split(content)

    current_section = "Introduction"
    current_text = ""

    def flush_chunk(text: str, section: str) -> None:
        nonlocal chunk_index
        text = text.strip()
        if not text:
            return

        token_count = count_tokens(text)

        if token_count <= max_tokens:
            chunks.append(
                Chunk(
                    chunk_id=hash_content(f"{document_id}-{chunk_index}-{text}"),
                    document_id=document_id,
                    content=text,
                    token_count=token_count,
                    chunk_index=chunk_index,
                    section_path=section,
                )
            )
            chunk_index += 1
        else:
            # Split large sections by paragraphs
            paragraphs = re.split(r"\n\n+", text)
            para_buffer = ""
            para_tokens = 0

            for para in paragraphs:
                para_count = count_tokens(para)

                if para_tokens + para_count > max_tokens and para_buffer:
                    chunks.append(
                        Chunk(
                            chunk_id=hash_content(f"{document_id}-{chunk_index}-{para_buffer}"),
                            document_id=document_id,
                            content=para_buffer.strip(),
                            token_count=para_tokens,
                            chunk_index=chunk_index,
                            section_path=section,
                        )
                    )
                    chunk_index += 1
                    para_buffer = ""
                    para_tokens = 0

                if para_buffer:
                    para_buffer += "\n\n"
                para_buffer += para
                para_tokens += para_count

            if para_buffer.strip():
                chunks.append(
                    Chunk(
                        chunk_id=hash_content(f"{document_id}-{chunk_index}-{para_buffer}"),
                        document_id=document_id,
                        content=para_buffer.strip(),
                        token_count=para_tokens,
                        chunk_index=chunk_index,
                        section_path=section,
                    )
                )
                chunk_index += 1

    for part in parts:
        if not part.strip():
            continue

        # Check if this is a heading
        if section_pattern.match(part):
            # Flush current content before starting new section
            if current_text.strip():
                flush_chunk(current_text, current_section)
                current_text = ""

            # Extract section name from heading
            current_section = part.strip().lstrip("#").strip()
            current_text = part + "\n\n"
        else:
            current_text += part

    # Flush remaining content
    if current_text.strip():
        flush_chunk(current_text, current_section)

    return chunks


async def ingest_consensus_specs() -> None:
    """Ingest consensus specs into the database."""
    loader = ConsensusSpecLoader()
    embedder = VoyageEmbedder()
    store = PgVectorStore()
    await store.connect()
    await store.initialize_schema()

    try:
        git_commit = loader.clone_or_update()
        specs = loader.load_all_specs()
        logger.info("found_specs", count=len(specs), git_commit=git_commit[:8])

        total_chunks = 0

        for spec in specs:
            document_id = f"consensus-spec-{spec.fork}-{spec.name}"

            # Chunk the markdown content
            chunks = chunk_markdown(spec.content, document_id, max_tokens=512)
            if not chunks:
                logger.warning("no_chunks_for_spec", spec=spec.name, fork=spec.fork)
                continue

            # Embed chunks
            embedded = embedder.embed_chunks(chunks)

            # Store document metadata
            await store.store_generic_document(
                document_id=document_id,
                document_type="consensus_spec",
                title=spec.title,
                source="ethereum/consensus-specs",
                raw_content=spec.content,
                git_commit=git_commit,
                metadata={
                    "fork": spec.fork,
                    "spec_name": spec.name,
                    "file_path": str(spec.file_path),
                },
            )

            # Store chunks
            await store.store_embedded_chunks(embedded, git_commit=git_commit)

            total_chunks += len(embedded)
            logger.info(
                "ingested_spec",
                fork=spec.fork,
                name=spec.name,
                chunks=len(embedded),
            )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        await store.close()

    logger.info(
        "consensus_specs_ingestion_complete",
        specs=len(specs),
        total_chunks=total_chunks,
    )


def main() -> None:
    asyncio.run(ingest_consensus_specs())


if __name__ == "__main__":
    main()
