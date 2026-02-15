"""Validated Generator - RAG generation with citation validation (Phase 2)."""

import asyncio
import os
import re
import uuid
from dataclasses import dataclass

import anthropic
import structlog

from ..config import DEFAULT_MODEL
from ..evidence.evidence_ledger import Claim, EvidenceLedger, SupportLevel
from ..evidence.evidence_span import EvidenceSpan
from ..retrieval.simple_retriever import RetrievalResult, SimpleRetriever
from ..validation.citation_enforcer import CitationEnforcer, ResponseVerifier
from ..validation.claim_decomposer import ClaimDecomposer, HybridDecomposer
from ..validation.nli_validator import CitationValidation, NLIValidator

logger = structlog.get_logger()


@dataclass
class ValidatedClaim:
    """A claim with its validation result."""

    claim: Claim
    validation: CitationValidation
    flagged: bool  # True if claim should be flagged in output


@dataclass
class ValidatedGenerationResult:
    """Result from generation with validation."""

    query: str
    response: str
    validated_response: str  # Response with flags/warnings
    evidence_ledger: EvidenceLedger
    validations: list[ValidatedClaim]
    retrieval: RetrievalResult
    model: str
    input_tokens: int
    output_tokens: int

    # Validation summary
    total_claims: int = 0
    supported_claims: int = 0
    weak_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0

    # Phase 7: Citation enforcement
    citation_compliant: bool = True
    uncited_factual_claims: int = 0

    @property
    def support_ratio(self) -> float:
        """Ratio of supported claims."""
        if self.total_claims == 0:
            return 1.0
        return self.supported_claims / self.total_claims

    @property
    def is_trustworthy(self) -> bool:
        """Whether the response is considered trustworthy."""
        return (
            self.contradicted_claims == 0 and self.support_ratio >= 0.7 and self.citation_compliant
        )


class ValidatedGenerator:
    """RAG generator with NLI-based citation validation (Phase 2).

    Generates responses, validates citations using NLI, and flags
    claims that aren't well-supported by evidence.
    """

    def __init__(
        self,
        retriever: SimpleRetriever,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_context_tokens: int = 8000,
        use_claim_decomposition: bool = True,
        nli_device: str | None = None,
    ):
        self.retriever = retriever
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Initialize validation components
        self.nli_validator = NLIValidator(device=nli_device, use_minimal_spans=True)
        self.use_decomposition = use_claim_decomposition

        if use_claim_decomposition:
            self.decomposer = HybridDecomposer(
                llm_decomposer=ClaimDecomposer(api_key=self.api_key, model=model)
            )
        else:
            self.decomposer = None

        # Phase 7: Citation enforcement and response verification
        self.citation_enforcer = CitationEnforcer()
        self.response_verifier = ResponseVerifier()

    async def generate(
        self,
        query: str,
        top_k: int = 10,
        validate: bool = True,
    ) -> ValidatedGenerationResult:
        """Generate an answer with citation validation.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            validate: Whether to run NLI validation

        Returns:
            ValidatedGenerationResult with validated answer
        """
        # Retrieve relevant chunks
        retrieval = await self.retriever.retrieve(query=query, top_k=top_k)

        # Format context with citation markers
        context, citation_map = self._format_context_with_markers(retrieval)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Generate response
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_response = response.content[0].text

        # Build evidence ledger
        evidence_ledger = self._build_evidence_ledger(
            query=query,
            response=raw_response,
            citation_map=citation_map,
            retrieval=retrieval,
        )

        # Validate claims if requested
        validations = []
        if validate:
            validations = await self._validate_all_claims(evidence_ledger)

        # Generate validated response with flags
        validated_response = self._format_validated_response(
            response=raw_response,
            validations=validations,
        )

        # Calculate summary statistics
        total_claims = len(validations)
        supported = sum(
            1
            for v in validations
            if v.validation.support_level in [SupportLevel.STRONG, SupportLevel.PARTIAL]
        )
        weak = sum(1 for v in validations if v.validation.support_level == SupportLevel.WEAK)
        unsupported = sum(1 for v in validations if v.validation.support_level == SupportLevel.NONE)
        contradicted = sum(
            1 for v in validations if v.validation.support_level == SupportLevel.CONTRADICTION
        )

        # Phase 7: Citation enforcement
        citation_result = self.citation_enforcer.enforce(evidence_ledger)

        # Phase 7: Verify response (remove contradicted, tag unsupported)
        if validate and validations:
            verified_response, _removed = self.response_verifier.verify_response(
                response=validated_response,
                ledger=evidence_ledger,
                validations=validations,
            )
            validated_response = verified_response

        logger.info(
            "generated_validated_response",
            query=query[:50],
            total_claims=total_claims,
            supported=supported,
            weak=weak,
            unsupported=unsupported,
            contradicted=contradicted,
            citation_compliant=citation_result.is_compliant,
            uncited_factual=citation_result.uncited_factual,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return ValidatedGenerationResult(
            query=query,
            response=raw_response,
            validated_response=validated_response,
            evidence_ledger=evidence_ledger,
            validations=validations,
            retrieval=retrieval,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_claims=total_claims,
            supported_claims=supported,
            weak_claims=weak,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            citation_compliant=citation_result.is_compliant,
            uncited_factual_claims=citation_result.uncited_factual,
        )

    def _format_context_with_markers(
        self,
        retrieval: RetrievalResult,
    ) -> tuple[str, dict]:
        """Format context with citation markers."""
        context_parts = []
        citation_map = {}

        for i, result in enumerate(retrieval.results):
            chunk = result.chunk
            marker = f"[{i + 1}]"
            citation_map[marker] = result

            doc_id = chunk.document_id.upper()
            section = f" - {chunk.section_path}" if chunk.section_path else ""

            context_parts.append(f"{marker} {doc_id}{section}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)
        return context, citation_map

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the generation prompt."""
        return f"""You are an expert on Ethereum protocol development. Answer the question based on the provided context from Ethereum Improvement Proposals (EIPs).

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. When making factual claims, cite the source using the marker (e.g., [1], [2])
3. If the context doesn't contain enough information, say so clearly
4. Don't make up information not found in the context
5. Be precise about numerical values and technical details

<context>
{context}
</context>

<question>
{query}
</question>

Provide a clear, accurate answer with citations to the relevant sources."""

    def _build_evidence_ledger(
        self,
        query: str,
        response: str,
        citation_map: dict,
        retrieval: RetrievalResult,
    ) -> EvidenceLedger:
        """Build evidence ledger from response."""
        response_id = str(uuid.uuid4())[:8]

        ledger = EvidenceLedger(
            response_id=response_id,
            query=query,
            response_text=response,
        )

        sentences = self._extract_sentences_with_citations(response)

        for i, (sentence, citations) in enumerate(sentences):
            if not sentence.strip():
                continue

            start_offset = response.find(sentence)
            end_offset = start_offset + len(sentence) if start_offset >= 0 else 0

            claim_type = "factual" if citations else "interpretive"

            claim_id = f"claim-{i + 1}"
            ledger.add_claim(
                claim_id=claim_id,
                claim_text=sentence,
                claim_type=claim_type,
                start_offset=start_offset,
                end_offset=end_offset,
            )

            for citation_marker in citations:
                if citation_marker in citation_map:
                    result = citation_map[citation_marker]
                    chunk = result.chunk

                    evidence = EvidenceSpan.from_chunk(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        section_path=chunk.section_path,
                    )
                    ledger.add_evidence(claim_id, evidence)

        return ledger

    def _extract_sentences_with_citations(
        self,
        response: str,
    ) -> list[tuple[str, list[str]]]:
        """Extract sentences and their citation markers."""
        citation_pattern = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
        sentence_pattern = re.compile(r"([^.!?]+[.!?]+)")

        sentences = sentence_pattern.findall(response)
        if not sentences:
            sentences = [response]

        results = []
        for sentence in sentences:
            matches = citation_pattern.findall(sentence)
            citations = []
            for match in matches:
                numbers = re.findall(r"\d+", match)
                for num in numbers:
                    citations.append(f"[{num}]")

            clean_sentence = citation_pattern.sub("", sentence).strip()
            results.append((clean_sentence, citations))

        return results

    async def _validate_all_claims(
        self,
        ledger: EvidenceLedger,
    ) -> list[ValidatedClaim]:
        """Validate all claims in the ledger."""
        validations = []

        for claim in ledger.claims:
            evidence_spans = ledger.get_evidence(claim.claim_id)

            # Optionally decompose complex claims
            atomic_facts = None
            if self.decomposer and claim.claim_type == "factual":
                atomic_facts = self.decomposer.decompose_sync(claim.claim_text)

            # Validate
            validation = self.nli_validator.validate_claim(
                claim=claim,
                evidence_spans=evidence_spans,
                atomic_facts=atomic_facts,
            )

            # Update claim with validation results
            claim.support_level = validation.support_level
            claim.confidence = validation.confidence
            claim.validation_method = validation.validation_method

            # Determine if claim should be flagged
            flagged = validation.support_level in [
                SupportLevel.WEAK,
                SupportLevel.NONE,
                SupportLevel.CONTRADICTION,
            ]

            validations.append(
                ValidatedClaim(
                    claim=claim,
                    validation=validation,
                    flagged=flagged,
                )
            )

        return validations

    def _format_validated_response(
        self,
        response: str,
        validations: list[ValidatedClaim],
    ) -> str:
        """Format response with validation flags."""
        if not validations:
            return response

        # Build annotations for flagged claims
        annotations = []

        for vc in validations:
            if vc.flagged:
                claim = vc.claim
                validation = vc.validation

                # Create warning annotation
                if validation.support_level == SupportLevel.CONTRADICTION:
                    warning = " ⚠️ [CONTRADICTED BY EVIDENCE]"
                elif validation.support_level == SupportLevel.NONE:
                    warning = " ⚠️ [NO SUPPORTING EVIDENCE]"
                elif validation.support_level == SupportLevel.WEAK:
                    warning = " ⚠️ [WEAKLY SUPPORTED]"
                else:
                    continue

                annotations.append((claim.end_offset, warning))

        # Apply annotations in reverse order
        result = response
        for offset, annotation in sorted(annotations, reverse=True):
            result = result[:offset] + annotation + result[offset:]

        # Add summary if there are issues
        flagged_count = sum(1 for v in validations if v.flagged)
        if flagged_count > 0:
            summary = f"\n\n---\n⚠️ **Validation Notice**: {flagged_count} claim(s) may have insufficient evidence support."
            result += summary

        return result

    def get_validation_report(
        self,
        result: ValidatedGenerationResult,
    ) -> str:
        """Generate a detailed validation report."""
        lines = [
            "# Citation Validation Report",
            "",
            f"**Query**: {result.query}",
            f"**Model**: {result.model}",
            "",
            "## Summary",
            f"- Total claims: {result.total_claims}",
            f"- Supported (strong/partial): {result.supported_claims}",
            f"- Weakly supported: {result.weak_claims}",
            f"- Unsupported: {result.unsupported_claims}",
            f"- Contradicted: {result.contradicted_claims}",
            f"- Support ratio: {result.support_ratio:.1%}",
            f"- Trustworthy: {'Yes' if result.is_trustworthy else 'No'}",
            "",
            "## Claim Details",
        ]

        for vc in result.validations:
            claim = vc.claim
            validation = vc.validation

            status_emoji = {
                SupportLevel.STRONG: "✅",
                SupportLevel.PARTIAL: "🟡",
                SupportLevel.WEAK: "⚠️",
                SupportLevel.NONE: "❌",
                SupportLevel.CONTRADICTION: "🚫",
            }.get(validation.support_level, "❓")

            lines.extend(
                [
                    "",
                    f"### {status_emoji} {claim.claim_id}",
                    f"**Claim**: {claim.claim_text}",
                    f"**Support Level**: {validation.support_level.value}",
                    f"**Confidence**: {validation.confidence:.2f}",
                    f"**Method**: {validation.validation_method}",
                ]
            )

            if validation.explanation:
                lines.append(f"**Explanation**: {validation.explanation}")

            if validation.atomic_results:
                lines.append("**Atomic Facts**:")
                for ar in validation.atomic_results:
                    fact_status = "✅" if ar.support_level == SupportLevel.STRONG else "⚠️"
                    lines.append(
                        f"  - {fact_status} {ar.fact.text} (entailment: {ar.nli_result.entailment:.2f})"
                    )

        return "\n".join(lines)
