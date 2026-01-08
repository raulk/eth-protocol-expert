"""Fixed Chunker - Simple fixed-size chunking for Phase 0."""

import hashlib
from dataclasses import dataclass

import structlog
import tiktoken

from ..ingestion.eip_parser import ParsedEIP

logger = structlog.get_logger()


@dataclass
class Chunk:
    """A chunk of text for embedding."""
    chunk_id: str
    document_id: str  # e.g., "eip-4844"
    content: str
    token_count: int
    chunk_index: int

    # For Phase 1+
    section_path: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None

    def __hash__(self):
        return hash(self.chunk_id)


class FixedChunker:
    """Simple fixed-size chunking (Phase 0)."""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_eip(self, parsed_eip: ParsedEIP) -> list[Chunk]:
        """Chunk an EIP into fixed-size pieces."""
        document_id = f"eip-{parsed_eip.eip_number}"

        # Create a clean text version (skip frontmatter)
        text = self._prepare_text(parsed_eip)

        # Tokenize
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.max_tokens:
            # Small document - single chunk
            return [Chunk(
                chunk_id=self._hash_content(text),
                document_id=document_id,
                content=text,
                token_count=len(tokens),
                chunk_index=0,
            )]

        # Split into overlapping chunks
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(Chunk(
                chunk_id=self._hash_content(chunk_text),
                document_id=document_id,
                content=chunk_text,
                token_count=len(chunk_tokens),
                chunk_index=chunk_index,
            ))

            chunk_index += 1
            start = end - self.overlap_tokens

            # Avoid infinite loop
            if start >= len(tokens) - self.overlap_tokens:
                break

        logger.debug(
            "chunked_eip",
            eip=parsed_eip.eip_number,
            chunks=len(chunks),
            total_tokens=len(tokens),
        )
        return chunks

    def _prepare_text(self, parsed_eip: ParsedEIP) -> str:
        """Prepare EIP content for chunking."""
        # Start with title and basic info
        parts = [
            f"# EIP-{parsed_eip.eip_number}: {parsed_eip.title}",
            f"Status: {parsed_eip.status}",
            f"Type: {parsed_eip.type}",
        ]

        if parsed_eip.category:
            parts.append(f"Category: {parsed_eip.category}")

        parts.append("")  # Blank line

        # Add all sections
        for section in parsed_eip.sections:
            parts.append(f"## {section.name}")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)

    def _hash_content(self, content: str) -> str:
        """Generate a content-based hash for chunk ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
