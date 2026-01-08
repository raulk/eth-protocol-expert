"""Evidence Ledger - Track claims and their supporting evidence (Phase 1)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .evidence_span import EvidenceSpan


class SupportLevel(Enum):
    """How strongly evidence supports a claim."""
    STRONG = "strong"       # NLI entailment > 0.7
    PARTIAL = "partial"     # NLI entailment 0.4-0.7
    WEAK = "weak"           # NLI entailment < 0.4
    NONE = "none"           # No supporting evidence
    CONTRADICTION = "contradiction"  # Evidence contradicts claim


@dataclass
class Claim:
    """A single factual assertion in a response."""
    claim_id: str
    claim_text: str
    claim_type: str           # factual | interpretive | synthetic

    # Position in response
    start_offset: int
    end_offset: int

    # Validation results (filled in Phase 2)
    support_level: SupportLevel | None = None
    confidence: float | None = None
    validation_method: str | None = None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "support_level": self.support_level.value if self.support_level else None,
            "confidence": self.confidence,
            "validation_method": self.validation_method,
        }


@dataclass
class ValidationResult:
    """Result of validating an evidence ledger."""
    is_valid: bool
    coverage_ratio: float     # Fraction of claims with evidence
    invalid_spans: list[tuple[str, EvidenceSpan]]  # (claim_id, span)
    unsupported_claims: list[str]  # claim_ids without evidence
    contradicted_claims: list[str]  # claim_ids with contradicting evidence


@dataclass
class EvidenceLedger:
    """Complete provenance tracking for a response.

    Maps every claim to its supporting evidence spans.
    """

    response_id: str
    query: str
    response_text: str

    claims: list[Claim] = field(default_factory=list)
    evidence_map: dict[str, list[EvidenceSpan]] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)
    validation_result: ValidationResult | None = None

    def add_claim(
        self,
        claim_id: str,
        claim_text: str,
        claim_type: str,
        start_offset: int,
        end_offset: int,
    ) -> Claim:
        """Add a claim to the ledger."""
        claim = Claim(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_type=claim_type,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        self.claims.append(claim)
        self.evidence_map[claim_id] = []
        return claim

    def add_evidence(self, claim_id: str, evidence: EvidenceSpan):
        """Add evidence supporting a claim."""
        if claim_id not in self.evidence_map:
            self.evidence_map[claim_id] = []
        self.evidence_map[claim_id].append(evidence)

    def get_evidence(self, claim_id: str) -> list[EvidenceSpan]:
        """Get all evidence for a claim."""
        return self.evidence_map.get(claim_id, [])

    def get_unsupported_claims(self) -> list[Claim]:
        """Get claims without any evidence."""
        return [
            claim for claim in self.claims
            if not self.evidence_map.get(claim.claim_id)
        ]

    def compute_coverage(self) -> float:
        """Compute fraction of claims with at least one evidence span."""
        if not self.claims:
            return 1.0
        supported = sum(
            1 for claim in self.claims
            if self.evidence_map.get(claim.claim_id)
        )
        return supported / len(self.claims)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "response_id": self.response_id,
            "query": self.query,
            "response_text": self.response_text,
            "claims": [c.to_dict() for c in self.claims],
            "evidence_map": {
                claim_id: [e.to_dict() for e in spans]
                for claim_id, spans in self.evidence_map.items()
            },
            "created_at": self.created_at.isoformat(),
            "coverage_ratio": self.compute_coverage(),
        }

    def format_response_with_citations(self) -> str:
        """Format the response with inline citations."""
        if not self.claims:
            return self.response_text

        # Build citation annotations
        annotations = []
        for claim in self.claims:
            evidence = self.evidence_map.get(claim.claim_id, [])
            if evidence:
                citations = " ".join(e.format_citation() for e in evidence)
                annotations.append((claim.end_offset, f" {citations}"))

        # Apply annotations in reverse order to preserve offsets
        result = self.response_text
        for offset, citation in sorted(annotations, reverse=True):
            result = result[:offset] + citation + result[offset:]

        return result

    def get_all_sources(self) -> list[dict]:
        """Get unique sources referenced in evidence."""
        sources = {}
        for spans in self.evidence_map.values():
            for span in spans:
                if span.document_id not in sources:
                    sources[span.document_id] = {
                        "document_id": span.document_id,
                        "sections": set(),
                    }
                if span.section_path:
                    sources[span.document_id]["sections"].add(span.section_path)

        return [
            {
                "document_id": s["document_id"],
                "sections": list(s["sections"]),
            }
            for s in sources.values()
        ]
