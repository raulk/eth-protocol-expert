"""Evidence Span - Track provenance of claims (Phase 1)."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EvidenceSpan:
    """Immutable reference to a specific text span in the corpus.

    This is the foundational unit of provenance. Every claim should
    map to one or more evidence spans.
    """

    # Document identification
    document_id: str  # Stable canonical ID (e.g., "eip-4844")
    chunk_id: str  # Chunk within document

    # Span location within chunk
    start_offset: int | None  # Character offset (None = whole chunk)
    end_offset: int | None  # Character offset (None = whole chunk)

    # Content snapshot
    span_text: str  # Exact text for validation
    span_hash: str  # SHA256 of span_text

    # Source metadata
    section_path: str | None  # e.g., "Motivation > Background"
    git_commit: str | None  # Revision hash for drift detection
    retrieved_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_chunk(
        cls,
        document_id: str,
        chunk_id: str,
        content: str,
        section_path: str | None = None,
        git_commit: str | None = None,
    ) -> "EvidenceSpan":
        """Create an evidence span from a whole chunk."""
        return cls(
            document_id=document_id,
            chunk_id=chunk_id,
            start_offset=None,
            end_offset=None,
            span_text=content,
            span_hash=hashlib.sha256(content.encode()).hexdigest(),
            section_path=section_path,
            git_commit=git_commit,
        )

    @classmethod
    def from_substring(
        cls,
        document_id: str,
        chunk_id: str,
        full_content: str,
        start_offset: int,
        end_offset: int,
        section_path: str | None = None,
        git_commit: str | None = None,
    ) -> "EvidenceSpan":
        """Create an evidence span from a substring within a chunk."""
        span_text = full_content[start_offset:end_offset]
        return cls(
            document_id=document_id,
            chunk_id=chunk_id,
            start_offset=start_offset,
            end_offset=end_offset,
            span_text=span_text,
            span_hash=hashlib.sha256(span_text.encode()).hexdigest(),
            section_path=section_path,
            git_commit=git_commit,
        )

    def validate_against(self, current_content: str) -> bool:
        """Verify the span still matches current corpus content."""
        if self.start_offset is not None and self.end_offset is not None:
            current_span = current_content[self.start_offset : self.end_offset]
        else:
            current_span = current_content

        current_hash = hashlib.sha256(current_span.encode()).hexdigest()
        return current_hash == self.span_hash

    def format_citation(self) -> str:
        """Format as a human-readable citation."""
        citation = self.document_id.upper()
        if self.section_path:
            citation = f"{citation}, {self.section_path}"
        return f"[{citation}]"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "span_text": self.span_text,
            "span_hash": self.span_hash,
            "section_path": self.section_path,
            "git_commit": self.git_commit,
            "retrieved_at": self.retrieved_at.isoformat(),
        }
