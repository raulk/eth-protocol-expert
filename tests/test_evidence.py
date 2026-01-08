"""Tests for evidence tracking."""

import pytest
from datetime import datetime

from src.evidence.evidence_span import EvidenceSpan
from src.evidence.evidence_ledger import EvidenceLedger, Claim, SupportLevel


class TestEvidenceSpan:
    """Tests for EvidenceSpan."""

    def test_from_chunk(self):
        """Test creating evidence span from a chunk."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-4844",
            chunk_id="abc123",
            content="This is test content.",
            section_path="Abstract",
            git_commit="def456",
        )

        assert span.document_id == "eip-4844"
        assert span.chunk_id == "abc123"
        assert span.span_text == "This is test content."
        assert span.section_path == "Abstract"
        assert span.git_commit == "def456"
        assert span.start_offset is None
        assert span.end_offset is None

    def test_from_substring(self):
        """Test creating evidence span from a substring."""
        full_content = "The quick brown fox jumps over the lazy dog."
        span = EvidenceSpan.from_substring(
            document_id="eip-1559",
            chunk_id="xyz789",
            full_content=full_content,
            start_offset=4,
            end_offset=19,
            section_path="Motivation",
        )

        assert span.span_text == "quick brown fox"
        assert span.start_offset == 4
        assert span.end_offset == 19

    def test_span_hash_deterministic(self):
        """Test that span hash is deterministic."""
        span1 = EvidenceSpan.from_chunk(
            document_id="eip-1",
            chunk_id="a",
            content="Same content",
        )
        span2 = EvidenceSpan.from_chunk(
            document_id="eip-2",
            chunk_id="b",
            content="Same content",
        )

        # Same content should have same hash
        assert span1.span_hash == span2.span_hash

    def test_validate_against_matching_content(self):
        """Test validation against matching content."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-1",
            chunk_id="a",
            content="Original content",
        )

        assert span.validate_against("Original content") is True

    def test_validate_against_changed_content(self):
        """Test validation against changed content."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-1",
            chunk_id="a",
            content="Original content",
        )

        assert span.validate_against("Modified content") is False

    def test_format_citation(self):
        """Test citation formatting."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-4844",
            chunk_id="a",
            content="content",
            section_path="Motivation",
        )

        citation = span.format_citation()
        assert citation == "[EIP-4844, Motivation]"

    def test_format_citation_no_section(self):
        """Test citation formatting without section."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-4844",
            chunk_id="a",
            content="content",
        )

        citation = span.format_citation()
        assert citation == "[EIP-4844]"

    def test_to_dict(self):
        """Test serialization to dict."""
        span = EvidenceSpan.from_chunk(
            document_id="eip-1559",
            chunk_id="abc",
            content="Test",
            section_path="Abstract",
            git_commit="xyz",
        )

        data = span.to_dict()
        assert data["document_id"] == "eip-1559"
        assert data["chunk_id"] == "abc"
        assert data["section_path"] == "Abstract"
        assert "retrieved_at" in data


class TestEvidenceLedger:
    """Tests for EvidenceLedger."""

    @pytest.fixture
    def ledger(self):
        """Create a test ledger."""
        return EvidenceLedger(
            response_id="resp-123",
            query="What is EIP-4844?",
            response_text="EIP-4844 introduces blob transactions. They reduce L2 costs.",
        )

    def test_add_claim(self, ledger):
        """Test adding a claim."""
        claim = ledger.add_claim(
            claim_id="claim-1",
            claim_text="EIP-4844 introduces blob transactions",
            claim_type="factual",
            start_offset=0,
            end_offset=37,
        )

        assert claim.claim_id == "claim-1"
        assert len(ledger.claims) == 1
        assert "claim-1" in ledger.evidence_map

    def test_add_evidence(self, ledger):
        """Test adding evidence to a claim."""
        ledger.add_claim(
            claim_id="claim-1",
            claim_text="Test claim",
            claim_type="factual",
            start_offset=0,
            end_offset=10,
        )

        evidence = EvidenceSpan.from_chunk(
            document_id="eip-4844",
            chunk_id="abc",
            content="EIP-4844 introduces blob transactions.",
        )

        ledger.add_evidence("claim-1", evidence)

        assert len(ledger.get_evidence("claim-1")) == 1

    def test_get_unsupported_claims(self, ledger):
        """Test getting claims without evidence."""
        ledger.add_claim("claim-1", "Supported claim", "factual", 0, 10)
        ledger.add_claim("claim-2", "Unsupported claim", "factual", 10, 20)

        evidence = EvidenceSpan.from_chunk("eip-1", "a", "content")
        ledger.add_evidence("claim-1", evidence)

        unsupported = ledger.get_unsupported_claims()
        assert len(unsupported) == 1
        assert unsupported[0].claim_id == "claim-2"

    def test_compute_coverage_all_supported(self, ledger):
        """Test coverage when all claims are supported."""
        ledger.add_claim("claim-1", "Claim 1", "factual", 0, 10)
        ledger.add_claim("claim-2", "Claim 2", "factual", 10, 20)

        ledger.add_evidence("claim-1", EvidenceSpan.from_chunk("eip-1", "a", "c1"))
        ledger.add_evidence("claim-2", EvidenceSpan.from_chunk("eip-2", "b", "c2"))

        assert ledger.compute_coverage() == 1.0

    def test_compute_coverage_partial(self, ledger):
        """Test coverage when some claims are unsupported."""
        ledger.add_claim("claim-1", "Claim 1", "factual", 0, 10)
        ledger.add_claim("claim-2", "Claim 2", "factual", 10, 20)

        ledger.add_evidence("claim-1", EvidenceSpan.from_chunk("eip-1", "a", "c1"))
        # claim-2 has no evidence

        assert ledger.compute_coverage() == 0.5

    def test_compute_coverage_empty(self):
        """Test coverage with no claims."""
        ledger = EvidenceLedger(
            response_id="r1",
            query="q",
            response_text="response",
        )
        assert ledger.compute_coverage() == 1.0

    def test_get_all_sources(self, ledger):
        """Test getting all unique sources."""
        ledger.add_claim("claim-1", "Claim 1", "factual", 0, 10)
        ledger.add_claim("claim-2", "Claim 2", "factual", 10, 20)

        ledger.add_evidence("claim-1", EvidenceSpan.from_chunk(
            "eip-4844", "a", "c1", section_path="Abstract"
        ))
        ledger.add_evidence("claim-1", EvidenceSpan.from_chunk(
            "eip-4844", "b", "c2", section_path="Motivation"
        ))
        ledger.add_evidence("claim-2", EvidenceSpan.from_chunk(
            "eip-1559", "c", "c3", section_path="Specification"
        ))

        sources = ledger.get_all_sources()
        assert len(sources) == 2

        doc_ids = {s["document_id"] for s in sources}
        assert doc_ids == {"eip-4844", "eip-1559"}

    def test_to_dict(self, ledger):
        """Test serialization to dict."""
        ledger.add_claim("claim-1", "Test claim", "factual", 0, 10)
        ledger.add_evidence("claim-1", EvidenceSpan.from_chunk("eip-1", "a", "c"))

        data = ledger.to_dict()
        assert data["response_id"] == "resp-123"
        assert data["query"] == "What is EIP-4844?"
        assert len(data["claims"]) == 1
        assert "claim-1" in data["evidence_map"]
        assert data["coverage_ratio"] == 1.0
