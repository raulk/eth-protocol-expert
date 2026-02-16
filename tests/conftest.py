"""Shared test fixtures for Ethereum Protocol Intelligence System."""

from dataclasses import dataclass
from datetime import datetime

import pytest


@dataclass
class MockChunk:
    """Mock chunk for testing."""
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    chunk_index: int
    section_path: str | None = None


@dataclass
class MockStoredChunk:
    """Mock stored chunk for testing."""
    id: int
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    chunk_index: int
    section_path: str | None
    embedding: list[float]
    created_at: datetime


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    chunk: MockStoredChunk
    similarity: float


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    now = datetime.now()
    return [
        MockStoredChunk(
            id=1,
            chunk_id="eip-1559-chunk-0",
            document_id="eip-1559",
            content="EIP-1559 introduces a base fee mechanism that adjusts dynamically based on block utilization.",
            token_count=20,
            chunk_index=0,
            section_path="Abstract",
            embedding=[0.1] * 1024,
            created_at=now,
        ),
        MockStoredChunk(
            id=2,
            chunk_id="eip-1559-chunk-1",
            document_id="eip-1559",
            content="The base fee is burned, reducing ETH supply. Users also pay a priority fee to validators.",
            token_count=22,
            chunk_index=1,
            section_path="Specification",
            embedding=[0.2] * 1024,
            created_at=now,
        ),
        MockStoredChunk(
            id=3,
            chunk_id="eip-4844-chunk-0",
            document_id="eip-4844",
            content="EIP-4844 introduces blob transactions that provide cheap data availability for rollups.",
            token_count=18,
            chunk_index=0,
            section_path="Abstract",
            embedding=[0.3] * 1024,
            created_at=now,
        ),
        MockStoredChunk(
            id=4,
            chunk_id="eip-4844-chunk-1",
            document_id="eip-4844",
            content="Blobs use a separate fee market with its own base fee that adjusts based on blob space utilization.",
            token_count=20,
            chunk_index=1,
            section_path="Specification",
            embedding=[0.4] * 1024,
            created_at=now,
        ),
    ]


@pytest.fixture
def sample_search_results(sample_chunks):
    """Create sample search results for testing."""
    return [
        MockSearchResult(chunk=sample_chunks[0], similarity=0.95),
        MockSearchResult(chunk=sample_chunks[1], similarity=0.88),
        MockSearchResult(chunk=sample_chunks[2], similarity=0.82),
        MockSearchResult(chunk=sample_chunks[3], similarity=0.75),
    ]
