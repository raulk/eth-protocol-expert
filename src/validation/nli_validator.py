"""NLI Validator - Validate citations using Natural Language Inference (Phase 2)."""

from dataclasses import dataclass

import numpy as np
import structlog

from ..evidence.evidence_ledger import Claim, SupportLevel
from ..evidence.evidence_span import EvidenceSpan

logger = structlog.get_logger()


@dataclass
class NLIResult:
    """Result from NLI model inference."""
    entailment: float    # Probability evidence entails claim
    contradiction: float  # Probability evidence contradicts claim
    neutral: float       # Probability of neither

    @property
    def predicted_label(self) -> str:
        scores = {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
        }
        return max(scores, key=scores.get)


@dataclass
class AtomicFact:
    """A single atomic, independently verifiable fact."""
    text: str
    source_claim: str


@dataclass
class AtomicFactValidation:
    """Validation result for an atomic fact."""
    fact: AtomicFact
    nli_result: NLIResult
    support_level: SupportLevel


@dataclass
class CitationValidation:
    """Complete validation result for a claim-evidence pair."""
    claim_id: str
    claim_text: str
    support_level: SupportLevel
    is_valid: bool
    confidence: float
    validation_method: str
    atomic_results: list[AtomicFactValidation]
    explanation: str | None = None


class NLIValidator:
    """Validate citations using NLI model.

    Uses multi-layer validation:
    1. Decompose claims into atomic facts (optional, via ClaimDecomposer)
    2. Check each fact against evidence via NLI
    3. Aggregate results to determine support level

    Default model is facebook/bart-large-mnli (public, no auth required).
    For higher accuracy, use microsoft/deberta-v3-large-mnli (requires HF login).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.7,
        weak_threshold: float = 0.4,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.weak_threshold = weak_threshold

        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._device = device

    def _load_model(self):
        """Lazy load the NLI model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("loading_nli_model", model=self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model.to(self._device)
        self._model.eval()

        logger.info("nli_model_loaded", device=self._device)

    def predict_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference.

        Args:
            premise: The evidence text (what we know is true)
            hypothesis: The claim to verify (what we're checking)

        Returns:
            NLIResult with entailment/contradiction/neutral scores
        """
        self._load_model()

        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Get label mapping from model config
        id2label = self._model.config.id2label
        label_to_idx = {v.lower(): k for k, v in id2label.items()}

        return NLIResult(
            contradiction=float(probs[label_to_idx.get("contradiction", 0)]),
            neutral=float(probs[label_to_idx.get("neutral", 1)]),
            entailment=float(probs[label_to_idx.get("entailment", 2)]),
        )

    def validate_claim(
        self,
        claim: Claim,
        evidence_spans: list[EvidenceSpan],
        atomic_facts: list[AtomicFact] | None = None,
    ) -> CitationValidation:
        """Validate a claim against its evidence.

        Args:
            claim: The claim to validate
            evidence_spans: Evidence that supposedly supports the claim
            atomic_facts: Optional pre-decomposed atomic facts

        Returns:
            CitationValidation with support level and details
        """
        if not evidence_spans:
            return CitationValidation(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                support_level=SupportLevel.NONE,
                is_valid=False,
                confidence=0.0,
                validation_method="no_evidence",
                atomic_results=[],
                explanation="No evidence spans provided for this claim.",
            )

        # Combine evidence texts
        evidence_text = "\n\n".join(span.span_text for span in evidence_spans)

        # If atomic facts provided, validate each
        if atomic_facts:
            return self._validate_atomic_facts(
                claim=claim,
                evidence_text=evidence_text,
                atomic_facts=atomic_facts,
            )

        # Otherwise validate claim directly
        return self._validate_single_claim(
            claim=claim,
            evidence_text=evidence_text,
        )

    def _validate_single_claim(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> CitationValidation:
        """Validate a single claim without decomposition."""
        nli_result = self.predict_nli(
            premise=evidence_text,
            hypothesis=claim.claim_text,
        )

        # Determine support level
        if nli_result.contradiction > self.contradiction_threshold:
            support_level = SupportLevel.CONTRADICTION
            is_valid = False
            explanation = "Evidence contradicts this claim."
        elif nli_result.entailment > self.entailment_threshold:
            support_level = SupportLevel.STRONG
            is_valid = True
            explanation = "Evidence strongly supports this claim."
        elif nli_result.entailment > self.weak_threshold:
            support_level = SupportLevel.PARTIAL
            is_valid = True
            explanation = "Evidence partially supports this claim."
        else:
            support_level = SupportLevel.WEAK
            is_valid = False
            explanation = "Evidence provides weak support for this claim."

        # Create a pseudo atomic fact for the whole claim
        atomic_fact = AtomicFact(text=claim.claim_text, source_claim=claim.claim_text)
        atomic_validation = AtomicFactValidation(
            fact=atomic_fact,
            nli_result=nli_result,
            support_level=support_level,
        )

        return CitationValidation(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            support_level=support_level,
            is_valid=is_valid,
            confidence=nli_result.entailment,
            validation_method="nli_direct",
            atomic_results=[atomic_validation],
            explanation=explanation,
        )

    def _validate_atomic_facts(
        self,
        claim: Claim,
        evidence_text: str,
        atomic_facts: list[AtomicFact],
    ) -> CitationValidation:
        """Validate claim by checking each atomic fact."""
        atomic_results = []

        for fact in atomic_facts:
            nli_result = self.predict_nli(
                premise=evidence_text,
                hypothesis=fact.text,
            )

            # Determine support for this fact
            if nli_result.contradiction > self.contradiction_threshold:
                support = SupportLevel.CONTRADICTION
            elif nli_result.entailment > self.entailment_threshold:
                support = SupportLevel.STRONG
            elif nli_result.entailment > self.weak_threshold:
                support = SupportLevel.PARTIAL
            else:
                support = SupportLevel.WEAK

            atomic_results.append(AtomicFactValidation(
                fact=fact,
                nli_result=nli_result,
                support_level=support,
            ))

        # Aggregate results
        return self._aggregate_atomic_results(claim, atomic_results)

    def _aggregate_atomic_results(
        self,
        claim: Claim,
        atomic_results: list[AtomicFactValidation],
    ) -> CitationValidation:
        """Aggregate atomic fact validations into claim validation."""
        if not atomic_results:
            return CitationValidation(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                support_level=SupportLevel.NONE,
                is_valid=False,
                confidence=0.0,
                validation_method="nli_atomic_empty",
                atomic_results=[],
            )

        # Check for any contradictions
        contradictions = [r for r in atomic_results if r.support_level == SupportLevel.CONTRADICTION]
        if contradictions:
            return CitationValidation(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                support_level=SupportLevel.CONTRADICTION,
                is_valid=False,
                confidence=0.0,
                validation_method="nli_atomic",
                atomic_results=atomic_results,
                explanation=f"{len(contradictions)} atomic fact(s) contradict the evidence.",
            )

        # Calculate average entailment
        avg_entailment = np.mean([r.nli_result.entailment for r in atomic_results])

        # Count support levels
        strong_count = sum(1 for r in atomic_results if r.support_level == SupportLevel.STRONG)
        partial_count = sum(1 for r in atomic_results if r.support_level == SupportLevel.PARTIAL)
        weak_count = sum(1 for r in atomic_results if r.support_level == SupportLevel.WEAK)

        # Determine overall support
        total = len(atomic_results)
        strong_ratio = strong_count / total

        if strong_ratio >= 0.8:
            support_level = SupportLevel.STRONG
            is_valid = True
            explanation = f"All atomic facts strongly supported ({strong_count}/{total})."
        elif strong_ratio >= 0.5 or (strong_count + partial_count) / total >= 0.8:
            support_level = SupportLevel.PARTIAL
            is_valid = True
            explanation = f"Most facts supported ({strong_count} strong, {partial_count} partial out of {total})."
        else:
            support_level = SupportLevel.WEAK
            is_valid = False
            explanation = f"Insufficient support ({weak_count} facts weakly supported out of {total})."

        return CitationValidation(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            support_level=support_level,
            is_valid=is_valid,
            confidence=float(avg_entailment),
            validation_method="nli_atomic",
            atomic_results=atomic_results,
            explanation=explanation,
        )

    def batch_validate(
        self,
        validations: list[tuple[Claim, list[EvidenceSpan]]],
    ) -> list[CitationValidation]:
        """Validate multiple claim-evidence pairs."""
        results = []
        for claim, evidence in validations:
            result = self.validate_claim(claim, evidence)
            results.append(result)

            logger.debug(
                "validated_claim",
                claim_id=claim.claim_id,
                support_level=result.support_level.value,
                confidence=result.confidence,
            )

        return results
