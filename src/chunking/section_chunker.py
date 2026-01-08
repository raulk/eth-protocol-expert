"""Section Chunker - Section-aware chunking for Phase 1."""

import hashlib
import re

import structlog
import tiktoken

from ..ingestion.eip_parser import EIPSection, ParsedEIP
from .fixed_chunker import Chunk

logger = structlog.get_logger()


class SectionChunker:
    """Section-aware chunking that respects EIP structure (Phase 1).

    Key improvements over FixedChunker:
    - Respects section boundaries (Abstract, Motivation, Specification, etc.)
    - Keeps code blocks atomic (never splits them)
    - Tracks section path for citations
    - Maintains document structure metadata
    """

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
        """Chunk an EIP respecting section boundaries and code blocks."""
        document_id = f"eip-{parsed_eip.eip_number}"
        chunks = []
        chunk_index = 0

        # Add a header chunk with metadata
        header_chunk = self._create_header_chunk(parsed_eip, document_id, chunk_index)
        chunks.append(header_chunk)
        chunk_index += 1

        # Process each section
        for section in parsed_eip.sections:
            section_chunks = self._chunk_section(
                section=section,
                document_id=document_id,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        logger.debug(
            "chunked_eip_by_section",
            eip=parsed_eip.eip_number,
            sections=len(parsed_eip.sections),
            chunks=len(chunks),
        )
        return chunks

    def _create_header_chunk(
        self, parsed_eip: ParsedEIP, document_id: str, chunk_index: int
    ) -> Chunk:
        """Create a header chunk with EIP metadata."""
        header_parts = [
            f"# EIP-{parsed_eip.eip_number}: {parsed_eip.title}",
            "",
            f"**Status**: {parsed_eip.status}",
            f"**Type**: {parsed_eip.type}",
        ]

        if parsed_eip.category:
            header_parts.append(f"**Category**: {parsed_eip.category}")
        if parsed_eip.author:
            header_parts.append(f"**Author**: {parsed_eip.author}")
        if parsed_eip.created:
            header_parts.append(f"**Created**: {parsed_eip.created}")
        if parsed_eip.requires:
            requires_str = ", ".join(f"EIP-{r}" for r in parsed_eip.requires)
            header_parts.append(f"**Requires**: {requires_str}")

        content = "\n".join(header_parts)
        return Chunk(
            chunk_id=self._hash_content(content),
            document_id=document_id,
            content=content,
            token_count=self.count_tokens(content),
            chunk_index=chunk_index,
            section_path="Header",
            start_offset=0,
            end_offset=len(content),
        )

    def _chunk_section(
        self,
        section: EIPSection,
        document_id: str,
        start_index: int,
    ) -> list[Chunk]:
        """Chunk a single section, keeping code blocks atomic."""
        chunks = []
        section_path = section.name

        # Extract code blocks and prose separately
        code_blocks, prose_parts = self._extract_code_blocks(section.content)

        current_index = start_index

        # First, chunk the prose
        for prose in prose_parts:
            if not prose.strip():
                continue

            prose_chunks = self._chunk_prose(
                prose=prose,
                document_id=document_id,
                section_path=section_path,
                start_index=current_index,
            )
            chunks.extend(prose_chunks)
            current_index += len(prose_chunks)

        # Then add code blocks as atomic chunks
        for i, code_block in enumerate(code_blocks):
            chunk_content = f"**{section_path} - Code Block {i + 1}**\n\n{code_block}"
            token_count = self.count_tokens(chunk_content)

            # If code block exceeds max_tokens, include it anyway (atomic)
            # but log a warning
            if token_count > self.max_tokens:
                logger.warning(
                    "oversized_code_block",
                    document_id=document_id,
                    section=section_path,
                    tokens=token_count,
                )

            chunks.append(Chunk(
                chunk_id=self._hash_content(chunk_content),
                document_id=document_id,
                content=chunk_content,
                token_count=token_count,
                chunk_index=current_index,
                section_path=f"{section_path} > Code Block {i + 1}",
                start_offset=None,  # Code blocks don't track offset
                end_offset=None,
            ))
            current_index += 1

        return chunks

    def _extract_code_blocks(self, content: str) -> tuple[list[str], list[str]]:
        """Extract code blocks and return (code_blocks, prose_parts)."""
        code_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)

        code_blocks = code_pattern.findall(content)
        prose_parts = code_pattern.split(content)

        return code_blocks, prose_parts

    def _chunk_prose(
        self,
        prose: str,
        document_id: str,
        section_path: str,
        start_index: int,
    ) -> list[Chunk]:
        """Chunk prose content, splitting at paragraph boundaries."""
        tokens = self.tokenizer.encode(prose)

        if len(tokens) <= self.max_tokens:
            # Fits in one chunk
            return [Chunk(
                chunk_id=self._hash_content(prose),
                document_id=document_id,
                content=prose.strip(),
                token_count=len(tokens),
                chunk_index=start_index,
                section_path=section_path,
                start_offset=0,
                end_offset=len(prose),
            )]

        # Split at paragraph boundaries
        paragraphs = re.split(r"\n\n+", prose)
        chunks = []
        current_paragraphs = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens > self.max_tokens and current_paragraphs:
                # Create chunk from accumulated paragraphs
                chunk_content = "\n\n".join(current_paragraphs)
                chunks.append(Chunk(
                    chunk_id=self._hash_content(chunk_content),
                    document_id=document_id,
                    content=chunk_content,
                    token_count=current_tokens,
                    chunk_index=start_index + len(chunks),
                    section_path=section_path,
                    start_offset=None,
                    end_offset=None,
                ))
                current_paragraphs = []
                current_tokens = 0

            current_paragraphs.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_paragraphs:
            chunk_content = "\n\n".join(current_paragraphs)
            chunks.append(Chunk(
                chunk_id=self._hash_content(chunk_content),
                document_id=document_id,
                content=chunk_content,
                token_count=current_tokens,
                chunk_index=start_index + len(chunks),
                section_path=section_path,
                start_offset=None,
                end_offset=None,
            ))

        return chunks

    def _hash_content(self, content: str) -> str:
        """Generate a content-based hash for chunk ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
