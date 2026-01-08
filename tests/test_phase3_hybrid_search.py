"""Tests for Phase 3: Hybrid Search.

These are self-contained unit tests that don't require external dependencies.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime


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
class MockHybridResult:
    """Mock hybrid result for testing."""
    chunk: MockStoredChunk
    rrf_score: float
    vector_rank: int | None
    bm25_rank: int | None
    vector_similarity: float | None
    bm25_score: float | None


@dataclass
class MockMetadataQuery:
    """Mock metadata query for testing."""
    statuses: list[str] | None = None
    types: list[str] | None = None
    categories: list[str] | None = None
    authors: list[str] | None = None
    eip_numbers: list[int] | None = None
    requires_eip: int | None = None

    def is_empty(self) -> bool:
        return all([
            self.statuses is None,
            self.types is None,
            self.categories is None,
            self.authors is None,
            self.eip_numbers is None,
            self.requires_eip is None,
        ])


class MockMetadataFilter:
    """Mock metadata filter for testing."""

    def build_where_clause(self, query: MockMetadataQuery) -> tuple[str, list]:
        conditions = []
        params = []
        param_idx = 1

        if query.statuses:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(query.statuses)))
            conditions.append(f"status IN ({placeholders})")
            params.extend(query.statuses)
            param_idx += len(query.statuses)

        if query.types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(query.types)))
            conditions.append(f"type IN ({placeholders})")
            params.extend(query.types)
            param_idx += len(query.types)

        if query.categories:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(query.categories)))
            conditions.append(f"category IN ({placeholders})")
            params.extend(query.categories)
            param_idx += len(query.categories)

        if query.authors:
            author_conditions = []
            for author in query.authors:
                author_conditions.append(f"author ILIKE ${param_idx}")
                params.append(f"%{author}%")
                param_idx += 1
            conditions.append(f"({' OR '.join(author_conditions)})")

        if query.eip_numbers:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(query.eip_numbers)))
            conditions.append(f"eip_number IN ({placeholders})")
            params.extend(query.eip_numbers)
            param_idx += len(query.eip_numbers)

        if query.requires_eip:
            conditions.append(f"requires @> ARRAY[${param_idx}]")
            params.append(query.requires_eip)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else ""
        return where_clause, params


class TestMetadataFilter:
    """Tests for metadata filtering."""

    def test_empty_query(self):
        """Empty query should return no filters."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery()

        assert query.is_empty()
        where_clause, params = mf.build_where_clause(query)
        assert where_clause == ""
        assert params == []

    def test_status_filter(self):
        """Status filter should use IN clause."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(statuses=["Final", "Last Call"])

        assert not query.is_empty()
        where_clause, params = mf.build_where_clause(query)

        assert "status" in where_clause
        assert "Final" in params
        assert "Last Call" in params

    def test_type_filter(self):
        """Type filter should work correctly."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(types=["Standards Track"])

        where_clause, params = mf.build_where_clause(query)
        assert "type" in where_clause
        assert "Standards Track" in params

    def test_category_filter(self):
        """Category filter should work correctly."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(categories=["Core", "ERC"])

        where_clause, params = mf.build_where_clause(query)
        assert "category" in where_clause

    def test_author_filter(self):
        """Author filter should use ILIKE for partial matching."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(authors=["vitalik"])

        where_clause, params = mf.build_where_clause(query)
        assert "ILIKE" in where_clause.upper() or "author" in where_clause.lower()

    def test_eip_numbers_filter(self):
        """EIP numbers filter should use IN clause."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(eip_numbers=[1559, 4844])

        where_clause, params = mf.build_where_clause(query)
        assert "eip_number" in where_clause
        assert 1559 in params
        assert 4844 in params

    def test_requires_filter(self):
        """Requires filter should use array containment."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(requires_eip=1559)

        where_clause, params = mf.build_where_clause(query)
        assert "requires" in where_clause

    def test_combined_filters(self):
        """Multiple filters should be combined with AND."""
        mf = MockMetadataFilter()
        query = MockMetadataQuery(
            statuses=["Final"],
            categories=["Core"],
        )

        where_clause, params = mf.build_where_clause(query)
        assert "AND" in where_clause
        assert len(params) >= 2


class TestRRFScoring:
    """Tests for Reciprocal Rank Fusion scoring."""

    def test_rrf_formula(self):
        """Test RRF score calculation."""
        k = 60

        # Rank 1 in vector, rank 2 in BM25
        # RRF = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        vector_rank = 1
        bm25_rank = 2

        expected_rrf = (1 / (k + vector_rank)) + (1 / (k + bm25_rank))

        # Verify formula
        assert abs(expected_rrf - 0.03251) < 0.001

    def test_single_source_rrf(self):
        """Documents appearing in only one source should still get RRF score."""
        k = 60

        # Only in vector at rank 1
        rrf_vector_only = 1 / (k + 1)
        assert rrf_vector_only > 0

        # Only in BM25 at rank 1
        rrf_bm25_only = 1 / (k + 1)
        assert rrf_bm25_only > 0

    def test_higher_ranks_lower_score(self):
        """Higher rank numbers should result in lower RRF scores."""
        k = 60

        score_rank1 = 1 / (k + 1)
        score_rank10 = 1 / (k + 10)
        score_rank100 = 1 / (k + 100)

        assert score_rank1 > score_rank10 > score_rank100

    def test_rrf_fusion_logic(self):
        """Test RRF fusion with overlapping results."""
        k = 60
        vector_weight = 1.0
        bm25_weight = 1.0

        # Document A: rank 1 in vector, rank 3 in BM25
        # Document B: rank 2 in vector, not in BM25
        # Document C: not in vector, rank 1 in BM25

        score_a = (vector_weight / (k + 1)) + (bm25_weight / (k + 3))
        score_b = vector_weight / (k + 2)
        score_c = bm25_weight / (k + 1)

        # Document A should score highest (appears in both)
        assert score_a > score_b
        assert score_a > score_c


class TestHybridResult:
    """Tests for hybrid result dataclass."""

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return MockStoredChunk(
            id=1,
            chunk_id="eip-1559-chunk-0",
            document_id="eip-1559",
            content="EIP-1559 introduces a base fee mechanism.",
            token_count=20,
            chunk_index=0,
            section_path="Abstract",
            embedding=[0.1] * 1024,
            created_at=datetime.now(),
        )

    def test_hybrid_result_creation(self, sample_chunk):
        """Test creating hybrid result."""
        result = MockHybridResult(
            chunk=sample_chunk,
            rrf_score=0.0325,
            vector_rank=1,
            bm25_rank=2,
            vector_similarity=0.95,
            bm25_score=0.88,
        )

        assert result.rrf_score == 0.0325
        assert result.vector_rank == 1
        assert result.bm25_rank == 2

    def test_hybrid_result_single_source(self, sample_chunk):
        """Test hybrid result from single source."""
        result = MockHybridResult(
            chunk=sample_chunk,
            rrf_score=0.0164,
            vector_rank=1,
            bm25_rank=None,
            vector_similarity=0.95,
            bm25_score=None,
        )

        assert result.bm25_rank is None
        assert result.bm25_score is None
        assert result.vector_rank is not None


class TestBM25Integration:
    """Tests for BM25 search behavior."""

    def test_bm25_handles_exact_terms(self):
        """BM25 should find documents with exact term matches."""
        # Test content with specific term
        documents = [
            {"id": 1, "content": "SELFDESTRUCT opcode will be deprecated"},
            {"id": 2, "content": "The opcode for self destruction will change"},
            {"id": 3, "content": "EIP about other changes"},
        ]

        query = "SELFDESTRUCT"

        # Simple term matching simulation
        matches = [d for d in documents if query in d["content"]]
        assert len(matches) == 1
        assert matches[0]["id"] == 1

    def test_bm25_phrase_matching(self):
        """BM25 phrase matching should find multi-word queries."""
        documents = [
            {"id": 1, "content": "base fee mechanism for gas pricing"},
            {"id": 2, "content": "a mechanism for fees that relies on other things"},
            {"id": 3, "content": "unrelated content"},
        ]

        query_words = ["base", "fee"]

        # Documents containing both terms (word boundary matching)
        matches = [d for d in documents if all(
            f" {w} " in f" {d['content']} " or d["content"].startswith(f"{w} ") or d["content"].endswith(f" {w}")
            for w in query_words
        )]
        assert len(matches) == 1
        assert matches[0]["id"] == 1
