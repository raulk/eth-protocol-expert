"""Utility to convert specialized chunk types to standard Chunk format."""

from typing import Any

from src.chunking.fixed_chunker import Chunk


def to_standard_chunk(
    chunk: Any,
    document_id: str,
) -> Chunk:
    """Convert a specialized chunk to standard Chunk format.

    Handles: PaperChunk, TranscriptChunk, CodeChunk, and any chunk with
    chunk_id, content, token_count, chunk_index attributes.
    """
    section_path = None
    for attr in ["section_path", "section", "speaker", "function_name"]:
        if hasattr(chunk, attr):
            value = getattr(chunk, attr)
            if value:
                section_path = value
                break

    chunk_id = getattr(chunk, "chunk_id", f"{document_id}-{getattr(chunk, 'chunk_index', 0)}")
    content = getattr(chunk, "content", "")
    token_count = getattr(chunk, "token_count", len(content.split()))
    chunk_index = getattr(chunk, "chunk_index", 0)

    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        token_count=token_count,
        chunk_index=chunk_index,
        section_path=section_path,
    )


def convert_chunks(
    chunks: list[Any],
    document_id: str,
) -> list[Chunk]:
    """Convert a list of specialized chunks to standard Chunks."""
    return [to_standard_chunk(c, document_id) for c in chunks]
