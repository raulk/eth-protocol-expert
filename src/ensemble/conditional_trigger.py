"""Conditional Trigger - Determine when to use ensemble models (Phase 12)."""

import re
from dataclasses import dataclass, field
from typing import ClassVar

import structlog

from src.config import MODEL_BALANCED, MODEL_FAST, MODEL_POWERFUL
from src.ensemble.confidence_scorer import RetrievalConfidence
from src.routing.query_classifier import ClassificationResult, QueryType

logger = structlog.get_logger()


@dataclass
class EnsembleDecision:
    """Decision on whether to use ensemble models."""

    use_ensemble: bool
    reason: str
    suggested_models: list[str] = field(default_factory=list)
    suggested_strategy: str = "best"
    priority: int = 0


class ConditionalTrigger:
    """Determine when ensemble models should be used.

    Trigger conditions:
    - Low retrieval confidence
    - High-stakes queries (roadmap, architecture decisions)
    - Explicit user request for thorough analysis
    - Conflicting information detected
    - Complex multi-hop queries
    """

    HIGH_STAKES_PATTERNS: ClassVar[list[str]] = [
        r"\broadmap\b",
        r"\barchitecture\b",
        r"\bdesign\s+decision\b",
        r"\bbreaking\s+change\b",
        r"\bsecurity\b",
        r"\bcritical\b",
        r"\bproduction\b",
        r"\bmainnet\b",
        r"\bupgrade\b",
        r"\bhard\s*fork\b",
        r"\bconsensus\s+change\b",
    ]

    THOROUGH_ANALYSIS_PATTERNS: ClassVar[list[str]] = [
        r"\bthorough(ly)?\b",
        r"\bcomprehensive\b",
        r"\bin[\-\s]?depth\b",
        r"\bdetailed\s+analysis\b",
        r"\bfull\s+picture\b",
        r"\ball\s+perspectives\b",
        r"\bexhaustive\b",
        r"\bcareful(ly)?\s+(analyze|examine|review)\b",
    ]

    CONFLICT_INDICATORS: ClassVar[list[str]] = [
        r"\bcontradicts?\b",
        r"\bconflicting\b",
        r"\bdisagree(ment)?\b",
        r"\binconsistent\b",
        r"\bdispute[ds]?\b",
        r"\bopposing\s+views?\b",
        r"\bdebate[ds]?\b",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        high_stakes_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.high_stakes_threshold = high_stakes_threshold

    def should_ensemble(
        self,
        query: str,
        initial_confidence: float | None = None,
        retrieval_confidence: RetrievalConfidence | None = None,
        classification: ClassificationResult | None = None,
        context_has_conflicts: bool = False,
    ) -> EnsembleDecision:
        """Determine whether to use ensemble models.

        Args:
            query: The user's query
            initial_confidence: Optional simple confidence score (0-1)
            retrieval_confidence: Optional detailed retrieval confidence
            classification: Optional query classification result
            context_has_conflicts: Whether retrieved context contains conflicting info

        Returns:
            EnsembleDecision with recommendation and reasoning
        """
        triggers: list[tuple[str, int, str]] = []

        confidence = initial_confidence
        if retrieval_confidence:
            confidence = retrieval_confidence.score

        if confidence is not None and confidence < self.confidence_threshold:
            priority = int((self.confidence_threshold - confidence) * 100)
            triggers.append(
                (
                    f"Low retrieval confidence ({confidence:.2f} < {self.confidence_threshold})",
                    priority,
                    "merge",
                )
            )

        high_stakes_score = self._calculate_high_stakes_score(query)
        if high_stakes_score >= self.high_stakes_threshold:
            priority = int(high_stakes_score * 50)
            triggers.append(
                (
                    f"High-stakes query detected (score: {high_stakes_score:.2f})",
                    priority,
                    "vote",
                )
            )

        if self._requests_thorough_analysis(query):
            triggers.append(
                (
                    "User requested thorough/comprehensive analysis",
                    40,
                    "merge",
                )
            )

        if context_has_conflicts or self._mentions_conflicts(query):
            triggers.append(
                (
                    "Conflicting information detected or queried",
                    45,
                    "vote",
                )
            )

        if classification:
            if classification.query_type == QueryType.MULTI_HOP:
                triggers.append(
                    (
                        f"Multi-hop query requiring {classification.estimated_sub_questions} sub-questions",
                        35,
                        "merge",
                    )
                )
            elif classification.query_type == QueryType.COMPLEX and classification.confidence < 0.7:
                triggers.append(
                    (
                        "Complex query with uncertain classification",
                        25,
                        "best",
                    )
                )

        if not triggers:
            return EnsembleDecision(
                use_ensemble=False,
                reason="Single model response should be sufficient",
                suggested_strategy="best",
            )

        triggers.sort(key=lambda x: x[1], reverse=True)
        primary_trigger = triggers[0]
        combined_reasons = "; ".join(t[0] for t in triggers[:3])

        strategy_votes = {}
        for _, priority, strategy in triggers:
            strategy_votes[strategy] = strategy_votes.get(strategy, 0) + priority

        suggested_strategy = max(strategy_votes.keys(), key=lambda s: strategy_votes[s])

        suggested_models = self._suggest_models(triggers, high_stakes_score)

        max_priority = primary_trigger[1]

        logger.info(
            "ensemble_decision",
            use_ensemble=True,
            triggers=[t[0] for t in triggers],
            priority=max_priority,
            strategy=suggested_strategy,
        )

        return EnsembleDecision(
            use_ensemble=True,
            reason=combined_reasons,
            suggested_models=suggested_models,
            suggested_strategy=suggested_strategy,
            priority=max_priority,
        )

    def should_escalate(
        self,
        current_response: str,
        current_confidence: float,
        query: str,
    ) -> EnsembleDecision:
        """Determine if we should escalate from single model to ensemble.

        Called after initial response to check if ensemble is warranted.
        """
        triggers: list[tuple[str, int]] = []

        if current_confidence < 0.5:
            triggers.append(
                (
                    f"Response confidence too low ({current_confidence:.2f})",
                    60,
                )
            )

        response_length = len(current_response.split())
        if response_length < 50:
            triggers.append(
                (
                    "Response too short for complex query",
                    40,
                )
            )

        uncertainty_phrases = [
            "I'm not sure",
            "unclear",
            "may be",
            "might be",
            "possibly",
            "uncertain",
            "not enough information",
        ]
        uncertainty_count = sum(
            1 for phrase in uncertainty_phrases if phrase.lower() in current_response.lower()
        )
        if uncertainty_count >= 2:
            triggers.append(
                (
                    f"High uncertainty in response ({uncertainty_count} indicators)",
                    50,
                )
            )

        high_stakes = self._calculate_high_stakes_score(query)
        if high_stakes >= self.high_stakes_threshold and current_confidence < 0.8:
            triggers.append(
                (
                    "High-stakes query with moderate confidence",
                    55,
                )
            )

        if not triggers:
            return EnsembleDecision(
                use_ensemble=False,
                reason="Current response is adequate",
            )

        triggers.sort(key=lambda x: x[1], reverse=True)
        combined_reasons = "; ".join(t[0] for t in triggers[:2])

        return EnsembleDecision(
            use_ensemble=True,
            reason=combined_reasons,
            suggested_models=[MODEL_BALANCED, MODEL_POWERFUL],
            suggested_strategy="merge",
            priority=triggers[0][1],
        )

    def _calculate_high_stakes_score(self, query: str) -> float:
        """Calculate how high-stakes the query is."""
        query_lower = query.lower()
        matches = sum(1 for pattern in self.HIGH_STAKES_PATTERNS if re.search(pattern, query_lower))
        return min(matches * 0.25, 1.0)

    def _requests_thorough_analysis(self, query: str) -> bool:
        """Check if user explicitly requests thorough analysis."""
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in self.THOROUGH_ANALYSIS_PATTERNS)

    def _mentions_conflicts(self, query: str) -> bool:
        """Check if query mentions conflicting information."""
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in self.CONFLICT_INDICATORS)

    def _suggest_models(
        self,
        triggers: list[tuple[str, int, str]],
        high_stakes_score: float,
    ) -> list[str]:
        """Suggest which models to use based on triggers."""
        max_priority = max(t[1] for t in triggers) if triggers else 0

        if max_priority >= 50 or high_stakes_score >= 0.75:
            return [MODEL_FAST, MODEL_BALANCED, MODEL_POWERFUL]

        if max_priority >= 35:
            return [MODEL_BALANCED, MODEL_POWERFUL]

        return [MODEL_FAST, MODEL_BALANCED]
