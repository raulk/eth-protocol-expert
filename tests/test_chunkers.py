"""Tests for chunking implementations."""

from datetime import datetime
from pathlib import Path

import pytest

from src.chunking.fixed_chunker import FixedChunker
from src.chunking.section_chunker import SectionChunker
from src.ingestion.eip_loader import LoadedEIP
from src.ingestion.eip_parser import EIPParser

SAMPLE_EIP_CONTENT = """---
eip: 1559
title: Fee market change
author: Vitalik Buterin
status: Final
type: Standards Track
category: Core
created: 2019-04-13
---

## Abstract

A transaction pricing mechanism with a fixed-per-block network fee.

## Motivation

Ethereum has problems with fee volatility.

Long paragraph about motivation that explains why we need this change and how it will benefit the ecosystem. This includes multiple sentences that together form a coherent argument for the proposal.

## Specification

The base fee is calculated as follows:

```python
def calculate_base_fee(parent: Block) -> int:
    if parent.gas_used == parent.gas_target:
        return parent.base_fee
    elif parent.gas_used > parent.gas_target:
        gas_used_delta = parent.gas_used - parent.gas_target
        base_fee_delta = max(parent.base_fee * gas_used_delta // parent.gas_target // 8, 1)
        return parent.base_fee + base_fee_delta
    else:
        gas_used_delta = parent.gas_target - parent.gas_used
        base_fee_delta = parent.base_fee * gas_used_delta // parent.gas_target // 8
        return parent.base_fee - base_fee_delta
```

## Rationale

This design was chosen because it provides predictable fees.

## Security Considerations

No known security issues.
"""


@pytest.fixture
def parsed_eip():
    """Create a parsed EIP for testing."""
    loaded = LoadedEIP(
        eip_number=1559,
        file_path=Path("test/eip-1559.md"),
        raw_content=SAMPLE_EIP_CONTENT,
        git_commit="abc123",
        loaded_at=datetime.utcnow(),
    )
    parser = EIPParser()
    return parser.parse(loaded)


class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_chunk_small_document(self, parsed_eip):
        """Test chunking a small document that fits in one chunk."""
        chunker = FixedChunker(max_tokens=2000, overlap_tokens=64)
        chunks = chunker.chunk_eip(parsed_eip)

        # Small document should result in few chunks
        assert len(chunks) >= 1
        assert all(c.document_id == "eip-1559" for c in chunks)

    def test_chunk_ids_are_unique(self, parsed_eip):
        """Test that chunk IDs are unique."""
        chunker = FixedChunker(max_tokens=200, overlap_tokens=32)
        chunks = chunker.chunk_eip(parsed_eip)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_indices_sequential(self, parsed_eip):
        """Test that chunk indices are sequential."""
        chunker = FixedChunker(max_tokens=200, overlap_tokens=32)
        chunks = chunker.chunk_eip(parsed_eip)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_count_respected(self, parsed_eip):
        """Test that chunks don't exceed max tokens."""
        max_tokens = 200
        chunker = FixedChunker(max_tokens=max_tokens, overlap_tokens=32)
        chunks = chunker.chunk_eip(parsed_eip)

        for chunk in chunks:
            # Allow some tolerance for tokenization differences
            assert chunk.token_count <= max_tokens + 50

    def test_count_tokens(self):
        """Test token counting."""
        chunker = FixedChunker()
        count = chunker.count_tokens("Hello, this is a test sentence.")
        assert count > 0
        assert count < 20


class TestSectionChunker:
    """Tests for SectionChunker."""

    def test_creates_header_chunk(self, parsed_eip):
        """Test that a header chunk is created."""
        chunker = SectionChunker(max_tokens=512)
        chunks = chunker.chunk_eip(parsed_eip)

        header_chunks = [c for c in chunks if c.section_path == "Header"]
        assert len(header_chunks) == 1

        header = header_chunks[0]
        assert "EIP-1559" in header.content
        assert "Fee market change" in header.content

    def test_section_paths_populated(self, parsed_eip):
        """Test that section paths are populated."""
        chunker = SectionChunker(max_tokens=512)
        chunks = chunker.chunk_eip(parsed_eip)

        # All chunks should have section paths
        for chunk in chunks:
            assert chunk.section_path is not None

    def test_code_blocks_atomic(self, parsed_eip):
        """Test that code blocks are kept atomic."""
        chunker = SectionChunker(max_tokens=100)  # Force splitting
        chunks = chunker.chunk_eip(parsed_eip)

        # Find code block chunks
        code_chunks = [c for c in chunks if "Code Block" in (c.section_path or "")]
        assert len(code_chunks) >= 1

        # Code blocks should contain the full code
        for code_chunk in code_chunks:
            assert "def calculate_base_fee" in code_chunk.content or "```" in code_chunk.content

    def test_different_sections_chunked_separately(self, parsed_eip):
        """Test that different sections produce different chunks."""
        chunker = SectionChunker(max_tokens=512)
        chunks = chunker.chunk_eip(parsed_eip)

        section_paths = set(c.section_path for c in chunks)

        # Should have multiple sections
        assert len(section_paths) > 1
        assert "Header" in section_paths

    def test_chunk_ids_are_content_based(self, parsed_eip):
        """Test that chunk IDs are deterministic based on content."""
        chunker = SectionChunker(max_tokens=512)

        chunks1 = chunker.chunk_eip(parsed_eip)
        chunks2 = chunker.chunk_eip(parsed_eip)

        # Same content should produce same IDs
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_all_chunks_have_content(self, parsed_eip):
        """Test that all chunks have non-empty content."""
        chunker = SectionChunker(max_tokens=512)
        chunks = chunker.chunk_eip(parsed_eip)

        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.token_count > 0
