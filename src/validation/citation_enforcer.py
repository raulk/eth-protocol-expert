"""Citation Enforcer - Ensure factual sentences have proper citations."""

import re
from dataclasses import dataclass
from typing import ClassVar

import structlog

from ..evidence.evidence_ledger import Claim, EvidenceLedger, SupportLevel

logger = structlog.get_logger()


@dataclass
class CitationIssue:
    """An issue with citation coverage."""

    claim_id: str
    sentence: str
    issue_type: str  # "uncited_factual", "weak_citation", "missing_evidence"
    severity: str  # "error", "warning", "info"
    suggestion: str | None = None


@dataclass
class CitationEnforcementResult:
    """Result of citation enforcement check."""

    is_compliant: bool
    total_sentences: int
    cited_sentences: int
    uncited_factual: int
    issues: list[CitationIssue]

    @property
    def citation_ratio(self) -> float:
        """Ratio of sentences with citations."""
        if self.total_sentences == 0:
            return 1.0
        return self.cited_sentences / self.total_sentences


class CitationEnforcer:
    """Enforce citation requirements on generated responses.

    Identifies factual sentences without citations and provides
    suggestions for improvement.
    """

    # Patterns that indicate factual claims requiring citation
    FACTUAL_PATTERNS: ClassVar[list[str]] = [
        r"\d+(?:\.\d+)?%",  # Percentages
        r"\$?\d+(?:,\d{3})*(?:\.\d+)?",  # Numbers/amounts
        r"EIP-\d+",  # EIP references
        r"ERC-\d+",  # ERC references
        r"(?:is|are|was|were)\s+(?:the|a|an)",  # Definitional statements
        r"(?:introduced|proposed|implemented)\s+(?:in|by)",  # Attribution
        r"(?:first|originally|initially)",  # Historical claims
        r"according\s+to",  # Source references
        r"(?:bytes?|bits?|gas)",  # Technical values
    ]

    # Patterns that indicate non-factual statements
    NON_FACTUAL_PATTERNS: ClassVar[list[str]] = [
        r"^(?:for example|e\.g\.|i\.e\.)",  # Examples
        r"^(?:in summary|to summarize)",  # Summaries
        r"^(?:this means|this suggests)",  # Interpretations
        r"(?:should|could|might|may)\s+be",  # Hedged statements
        r"^(?:let me|i will)",  # Meta-statements
    ]

    def __init__(
        self,
        min_sentence_length: int = 20,
        require_citation_ratio: float = 0.7,
    ):
        """Initialize citation enforcer.

        Args:
            min_sentence_length: Minimum characters for a sentence to require citation
            require_citation_ratio: Required ratio of cited factual sentences
        """
        self.min_sentence_length = min_sentence_length
        self.require_citation_ratio = require_citation_ratio

        # Compile patterns
        self.factual_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self.non_factual_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.NON_FACTUAL_PATTERNS
        ]

    def enforce(
        self,
        ledger: EvidenceLedger,
    ) -> CitationEnforcementResult:
        """Check citation compliance for an evidence ledger.

        Args:
            ledger: Evidence ledger with claims and evidence mappings

        Returns:
            CitationEnforcementResult with compliance status and issues
        """
        issues = []
        cited_count = 0
        factual_uncited = 0

        for claim in ledger.claims:
            evidence = ledger.get_evidence(claim.claim_id)

            # Check if claim has any evidence
            has_citation = len(evidence) > 0

            if has_citation:
                cited_count += 1
            elif self._is_factual_claim(claim.claim_text):
                factual_uncited += 1
                issues.append(
                    CitationIssue(
                        claim_id=claim.claim_id,
                        sentence=claim.claim_text[:100],
                        issue_type="uncited_factual",
                        severity="warning",
                        suggestion="Add citation to support this factual claim",
                    )
                )

            # Check for weak citations (low support level)
            if has_citation and claim.support_level == SupportLevel.WEAK:
                issues.append(
                    CitationIssue(
                        claim_id=claim.claim_id,
                        sentence=claim.claim_text[:100],
                        issue_type="weak_citation",
                        severity="info",
                        suggestion="Evidence weakly supports this claim - consider stronger sources",
                    )
                )

        # Calculate compliance
        total = len(ledger.claims)
        ratio = cited_count / total if total > 0 else 1.0
        is_compliant = (
            ratio >= self.require_citation_ratio
            and factual_uncited <= total * 0.1  # Max 10% uncited factual claims
        )

        logger.debug(
            "citation_enforcement",
            total_claims=total,
            cited=cited_count,
            uncited_factual=factual_uncited,
            is_compliant=is_compliant,
        )

        return CitationEnforcementResult(
            is_compliant=is_compliant,
            total_sentences=total,
            cited_sentences=cited_count,
            uncited_factual=factual_uncited,
            issues=issues,
        )

    def _is_factual_claim(self, text: str) -> bool:
        """Determine if a sentence appears to be a factual claim."""
        if len(text) < self.min_sentence_length:
            return False

        # Check for non-factual patterns first
        for pattern in self.non_factual_patterns:
            if pattern.search(text):
                return False

        # Check for factual patterns
        for pattern in self.factual_patterns:
            if pattern.search(text):
                return True

        # Default: sentences with technical terms are likely factual
        technical_terms = [
            "gas",
            "wei",
            "gwei",
            "block",
            "transaction",
            "contract",
            "validator",
            "beacon",
            "slot",
            "epoch",
            "shard",
        ]
        text_lower = text.lower()
        for term in technical_terms:
            if term in text_lower:
                return True

        return False

    def suggest_citations(
        self,
        uncited_claim: Claim,
        available_sources: list[dict],
    ) -> list[str]:
        """Suggest potential citations for an uncited claim.

        Args:
            uncited_claim: The claim needing citations
            available_sources: Available sources from retrieval

        Returns:
            List of suggested source IDs
        """
        # Extract EIP/ERC references from claim
        eip_refs = re.findall(r"EIP-(\d+)", uncited_claim.claim_text, re.IGNORECASE)
        erc_refs = re.findall(r"ERC-(\d+)", uncited_claim.claim_text, re.IGNORECASE)

        suggestions = []

        # Suggest sources that match EIP/ERC references
        for source in available_sources:
            doc_id = source.get("document_id", "").lower()

            for eip_num in eip_refs:
                if f"eip-{eip_num}" in doc_id:
                    suggestions.append(doc_id)

            for erc_num in erc_refs:
                if f"erc-{erc_num}" in doc_id:
                    suggestions.append(doc_id)

        return list(set(suggestions))


class ResponseVerifier:
    """Verify and optionally modify responses to remove/tag unsupported claims."""

    def __init__(
        self,
        remove_contradicted: bool = True,
        tag_unsupported: bool = True,
        min_support_ratio: float = 0.5,
    ):
        """Initialize response verifier.

        Args:
            remove_contradicted: Whether to remove contradicted claims
            tag_unsupported: Whether to tag unsupported claims
            min_support_ratio: Minimum support ratio for a claim to be included
        """
        self.remove_contradicted = remove_contradicted
        self.tag_unsupported = tag_unsupported
        self.min_support_ratio = min_support_ratio

    def verify_response(
        self,
        response: str,
        ledger: EvidenceLedger,
        validations: list,
    ) -> tuple[str, list[str]]:
        """Verify and clean up a response.

        Args:
            response: Original response text
            ledger: Evidence ledger
            validations: List of ValidatedClaim from validation

        Returns:
            Tuple of (verified_response, list_of_removed_claims)
        """
        removed_claims = []
        modifications = []  # List of (start, end, replacement)

        for vc in validations:
            claim = vc.claim
            support_level = vc.validation.support_level

            if support_level == SupportLevel.CONTRADICTION and self.remove_contradicted:
                # Mark for removal
                removed_claims.append(claim.claim_text)
                modifications.append(
                    (
                        claim.start_offset,
                        claim.end_offset,
                        "",  # Remove entirely
                    )
                )

            elif support_level == SupportLevel.NONE and self.tag_unsupported:
                # Tag as unverified
                tag = " [unverified]"
                modifications.append(
                    (
                        claim.end_offset,
                        claim.end_offset,
                        tag,
                    )
                )

        # Apply modifications in reverse order to preserve offsets
        verified = response
        for start, end, replacement in sorted(modifications, reverse=True):
            verified = verified[:start] + replacement + verified[end:]

        # Clean up double spaces and empty sentences
        verified = re.sub(r"\s+", " ", verified)
        verified = re.sub(r"\.\s*\.", ".", verified)

        logger.info(
            "verified_response",
            original_length=len(response),
            verified_length=len(verified),
            removed_count=len(removed_claims),
        )

        return verified, removed_claims

    def generate_verification_summary(
        self,
        removed_claims: list[str],
        issues: list[CitationIssue],
    ) -> str:
        """Generate a summary of verification actions."""
        lines = []

        if removed_claims:
            lines.append(
                f"**Removed Claims ({len(removed_claims)})**: Claims contradicted by evidence were removed."
            )

        if issues:
            warnings = [i for i in issues if i.severity == "warning"]
            if warnings:
                lines.append(
                    f"**Citation Warnings ({len(warnings)})**: Some factual claims lack citations."
                )

        if not lines:
            lines.append("**Verification**: All claims are supported by cited evidence.")

        return "\n".join(lines)
