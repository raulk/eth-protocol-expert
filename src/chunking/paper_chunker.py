"""Paper Chunker - Section-aware chunking for academic papers."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import ClassVar

import structlog
import tiktoken

from src.ingestion.pdf_extractor import PDFContent, PDFSection

logger = structlog.get_logger()


@dataclass
class PaperChunk:
    """A chunk from an academic paper."""

    chunk_id: str
    content: str
    section: str
    paper_id: str
    page_numbers: list[int] = field(default_factory=list)
    token_count: int = 0
    chunk_index: int = 0
    has_equations: bool = False
    has_figures: bool = False


class PaperChunker:
    """Section-aware chunking for academic papers.

    Respects section boundaries and keeps equations, figures, and tables
    with their surrounding context when possible.
    """

    # Patterns that indicate equations or figures
    EQUATION_PATTERNS: ClassVar[list[str]] = [
        r"\$\$.*?\$\$",  # LaTeX display math
        r"\$[^$]+\$",  # LaTeX inline math
        r"\\begin\{equation\}.*?\\end\{equation\}",
        r"\\begin\{align\}.*?\\end\{align\}",
        r"=\s*\d+",  # Simple equations
    ]

    FIGURE_PATTERNS: ClassVar[list[str]] = [
        r"Figure\s+\d+",
        r"Fig\.\s*\d+",
        r"Table\s+\d+",
        r"\\begin\{figure\}",
        r"\\begin\{table\}",
    ]

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk(self, paper: PDFContent, max_tokens: int | None = None) -> list[PaperChunk]:
        """Chunk a paper respecting section boundaries.

        Args:
            paper: Extracted PDF content
            max_tokens: Override default max tokens per chunk

        Returns:
            List of PaperChunk objects
        """
        max_tok = max_tokens or self.max_tokens
        chunks = []
        chunk_index = 0

        # Create paper ID from title
        paper_id = self._create_paper_id(paper.title)

        # Add a header chunk with paper metadata
        header_chunk = self._create_header_chunk(paper, paper_id, chunk_index)
        chunks.append(header_chunk)
        chunk_index += 1

        # Process each section
        for section in paper.sections:
            section_chunks = self._chunk_section(
                section=section,
                paper_id=paper_id,
                start_index=chunk_index,
                max_tokens=max_tok,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        logger.debug(
            "chunked_paper",
            title=paper.title[:50] if paper.title else "untitled",
            sections=len(paper.sections),
            chunks=len(chunks),
        )
        return chunks

    def _create_header_chunk(
        self, paper: PDFContent, paper_id: str, chunk_index: int
    ) -> PaperChunk:
        """Create a header chunk with paper metadata."""
        parts = [f"# {paper.title}"]

        if paper.metadata.get("author"):
            parts.append(f"\n**Authors:** {paper.metadata['author']}")

        # Add abstract if we have it
        abstract = next(
            (s for s in paper.sections if s.heading.lower() == "abstract"),
            None,
        )
        if abstract and abstract.content:
            parts.append(f"\n**Abstract:** {abstract.content[:500]}")
            if len(abstract.content) > 500:
                parts.append("...")

        content = "\n".join(parts)

        return PaperChunk(
            chunk_id=self._hash_content(content),
            content=content,
            section="Header",
            paper_id=paper_id,
            page_numbers=[1],
            token_count=self.count_tokens(content),
            chunk_index=chunk_index,
        )

    def _chunk_section(
        self,
        section: PDFSection,
        paper_id: str,
        start_index: int,
        max_tokens: int,
    ) -> list[PaperChunk]:
        """Chunk a single section, keeping equations and figures with context."""
        chunks = []

        # Skip abstract in sections (already in header)
        if section.heading.lower() == "abstract":
            return chunks

        # Check for special content
        has_equations = self._has_equations(section.content)
        has_figures = self._has_figures(section.content)

        # Format section header
        section_header = f"## {section.heading}\n\n"
        header_tokens = self.count_tokens(section_header)

        # Available tokens for content
        content_max = max_tokens - header_tokens

        # Split content intelligently
        content_chunks = self._split_content(
            section.content, content_max, has_equations or has_figures
        )

        for i, content in enumerate(content_chunks):
            full_content = section_header + content

            chunks.append(
                PaperChunk(
                    chunk_id=self._hash_content(full_content),
                    content=full_content,
                    section=section.heading,
                    paper_id=paper_id,
                    page_numbers=section.page_numbers,
                    token_count=self.count_tokens(full_content),
                    chunk_index=start_index + i,
                    has_equations=self._has_equations(content),
                    has_figures=self._has_figures(content),
                )
            )

        return chunks

    def _split_content(self, content: str, max_tokens: int, preserve_special: bool) -> list[str]:
        """Split content into chunks, optionally preserving special elements."""
        if not content.strip():
            return []

        tokens = self.tokenizer.encode(content)

        if len(tokens) <= max_tokens:
            return [content.strip()]

        if preserve_special:
            # Try to split at paragraph boundaries while keeping equations/figures
            return self._split_preserving_special(content, max_tokens)
        else:
            # Simple paragraph-based splitting
            return self._split_by_paragraphs(content, max_tokens)

    def _split_preserving_special(self, content: str, max_tokens: int) -> list[str]:
        """Split content while trying to keep equations/figures with context."""
        chunks = []

        # Find all special elements (equations, figures)
        special_positions = []
        for pattern in self.EQUATION_PATTERNS + self.FIGURE_PATTERNS:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                special_positions.append((match.start(), match.end()))

        # Sort by position
        special_positions.sort()

        # Split into paragraphs
        paragraphs = re.split(r"\n\n+", content)
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If paragraph alone exceeds max, split it further
            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split oversized paragraph
                sub_chunks = self._split_paragraph(para, max_tokens)
                chunks.extend(sub_chunks[:-1])

                # Keep last sub-chunk for potential merging
                if sub_chunks:
                    current_chunk = [sub_chunks[-1]]
                    current_tokens = self.count_tokens(sub_chunks[-1])
                continue

            if current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _split_by_paragraphs(self, content: str, max_tokens: int) -> list[str]:
        """Split content by paragraph boundaries."""
        chunks = []
        paragraphs = re.split(r"\n\n+", content)
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                sub_chunks = self._split_paragraph(para, max_tokens)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _split_paragraph(self, paragraph: str, max_tokens: int) -> list[str]:
        """Split a single oversized paragraph."""
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Token-level split for extremely long sentences
                tokens = self.tokenizer.encode(sentence)
                for i in range(0, len(tokens), max_tokens - self.overlap_tokens):
                    chunk_tokens = tokens[i : i + max_tokens]
                    chunks.append(self.tokenizer.decode(chunk_tokens))
                continue

            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _has_equations(self, content: str) -> bool:
        """Check if content contains equations."""
        for pattern in self.EQUATION_PATTERNS:
            if re.search(pattern, content, re.DOTALL):
                return True
        return False

    def _has_figures(self, content: str) -> bool:
        """Check if content references figures or tables."""
        for pattern in self.FIGURE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _create_paper_id(self, title: str) -> str:
        """Create a paper ID from the title."""
        # Simple slug from title
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        slug = slug[:50]  # Limit length
        return f"paper-{slug}" if slug else "paper-untitled"

    def _hash_content(self, content: str) -> str:
        """Generate a content-based hash for chunk ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
