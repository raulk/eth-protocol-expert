"""Circuit Breaker - Halt generation for low-evidence cases (Phase 9)."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import structlog

from src.ensemble.confidence_scorer import RetrievalConfidence

logger = structlog.get_logger()


class CircuitState(Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Halted, refusing requests
    HALF_OPEN = "half_open"  # Testing if recovery possible


class BreakReason(Enum):
    """Reasons for opening the circuit."""

    LOW_EVIDENCE = "low_evidence"
    NO_RELEVANT_RESULTS = "no_relevant_results"
    HIGH_UNCERTAINTY = "high_uncertainty"
    CONFLICTING_SOURCES = "conflicting_sources"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"


@dataclass
class CircuitDecision:
    """Decision from the circuit breaker."""

    allow: bool
    state: CircuitState
    reason: BreakReason | None = None
    message: str = ""
    suggested_action: str = ""
    confidence_gap: float = 0.0


class CircuitBreaker:
    """Circuit breaker for low-evidence generation protection.

    Prevents the system from generating potentially unreliable responses
    when evidence quality is insufficient. This protects against:
    - Hallucination from sparse retrieval
    - Overconfident answers on uncertain topics
    - Wasted compute on unanswerble queries
    """

    MINIMUM_EVIDENCE_THRESHOLD: ClassVar[float] = 0.4
    MINIMUM_RELEVANCE_THRESHOLD: ClassVar[float] = 0.5
    MINIMUM_RESULT_COUNT: ClassVar[int] = 2

    def __init__(
        self,
        evidence_threshold: float = 0.4,
        relevance_threshold: float = 0.5,
        min_results: int = 2,
        consecutive_failures_to_open: int = 3,
    ):
        self.evidence_threshold = evidence_threshold
        self.relevance_threshold = relevance_threshold
        self.min_results = min_results
        self.consecutive_failures_to_open = consecutive_failures_to_open

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._total_trips = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def check(
        self,
        retrieval_confidence: RetrievalConfidence | None = None,
        result_count: int | None = None,
        top_relevance: float | None = None,
        has_conflicting_sources: bool = False,
    ) -> CircuitDecision:
        """Check if generation should proceed.

        Args:
            retrieval_confidence: Full retrieval confidence assessment
            result_count: Number of retrieved results
            top_relevance: Highest relevance score among results
            has_conflicting_sources: Whether sources conflict significantly

        Returns:
            CircuitDecision indicating whether to proceed
        """
        if self._state == CircuitState.OPEN:
            return CircuitDecision(
                allow=False,
                state=CircuitState.OPEN,
                reason=BreakReason.RATE_LIMIT,
                message="Circuit is open due to repeated failures",
                suggested_action="Wait before retrying or reformulate query",
            )

        reasons: list[tuple[BreakReason, str, float]] = []

        if retrieval_confidence:
            if retrieval_confidence.score < self.evidence_threshold:
                gap = self.evidence_threshold - retrieval_confidence.score
                reasons.append(
                    (
                        BreakReason.LOW_EVIDENCE,
                        f"Evidence quality too low ({retrieval_confidence.score:.2f} < {self.evidence_threshold})",
                        gap,
                    )
                )

            if not retrieval_confidence.is_sufficient:
                reasons.append(
                    (
                        BreakReason.NO_RELEVANT_RESULTS,
                        "Retrieved results insufficient for reliable answer",
                        0.3,
                    )
                )

        if result_count is not None and result_count < self.min_results:
            reasons.append(
                (
                    BreakReason.NO_RELEVANT_RESULTS,
                    f"Too few results ({result_count} < {self.min_results})",
                    0.5,
                )
            )

        if top_relevance is not None and top_relevance < self.relevance_threshold:
            gap = self.relevance_threshold - top_relevance
            reasons.append(
                (
                    BreakReason.NO_RELEVANT_RESULTS,
                    f"No highly relevant results (best: {top_relevance:.2f} < {self.relevance_threshold})",
                    gap,
                )
            )

        if has_conflicting_sources:
            reasons.append(
                (
                    BreakReason.CONFLICTING_SOURCES,
                    "Sources contain significant conflicting information",
                    0.2,
                )
            )

        if not reasons:
            self._consecutive_failures = 0
            return CircuitDecision(
                allow=True,
                state=CircuitState.CLOSED,
                message="Evidence quality sufficient for generation",
            )

        self._consecutive_failures += 1

        if self._consecutive_failures >= self.consecutive_failures_to_open:
            self._state = CircuitState.OPEN
            self._total_trips += 1

        primary_reason = max(reasons, key=lambda x: x[2])
        combined_message = "; ".join(r[1] for r in reasons)

        suggested_action = self._suggest_action(primary_reason[0])

        logger.warning(
            "circuit_breaker_triggered",
            reason=primary_reason[0].value,
            consecutive_failures=self._consecutive_failures,
            state=self._state.value,
        )

        return CircuitDecision(
            allow=False,
            state=self._state,
            reason=primary_reason[0],
            message=combined_message,
            suggested_action=suggested_action,
            confidence_gap=primary_reason[2],
        )

    def check_simple(
        self,
        confidence_score: float,
        result_count: int,
    ) -> CircuitDecision:
        """Simplified check with just confidence and count.

        Args:
            confidence_score: Overall confidence (0-1)
            result_count: Number of results

        Returns:
            CircuitDecision
        """
        if confidence_score >= self.evidence_threshold and result_count >= self.min_results:
            self._consecutive_failures = 0
            return CircuitDecision(
                allow=True,
                state=CircuitState.CLOSED,
                message="Evidence quality sufficient",
            )

        reasons = []
        max_gap = 0.0

        if confidence_score < self.evidence_threshold:
            gap = self.evidence_threshold - confidence_score
            max_gap = max(max_gap, gap)
            reasons.append(f"Low confidence ({confidence_score:.2f})")

        if result_count < self.min_results:
            reasons.append(f"Insufficient results ({result_count})")
            max_gap = max(max_gap, 0.3)

        self._consecutive_failures += 1

        if self._consecutive_failures >= self.consecutive_failures_to_open:
            self._state = CircuitState.OPEN
            self._total_trips += 1

        return CircuitDecision(
            allow=False,
            state=self._state,
            reason=BreakReason.LOW_EVIDENCE,
            message="; ".join(reasons),
            suggested_action="Reformulate query with more specific terms",
            confidence_gap=max_gap,
        )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        previous_state = self._state
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        logger.info("circuit_breaker_reset", previous_state=previous_state.value)

    def half_open(self) -> None:
        """Set circuit to half-open state for testing recovery."""
        if self._state == CircuitState.OPEN:
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_breaker_half_open")

    def record_success(self) -> None:
        """Record a successful operation (for half-open state)."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            logger.info("circuit_breaker_recovered")
        else:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.consecutive_failures_to_open:
            self._state = CircuitState.OPEN
            self._total_trips += 1
            logger.warning(
                "circuit_breaker_opened",
                consecutive_failures=self._consecutive_failures,
                total_trips=self._total_trips,
            )

    def get_stats(self) -> dict[str, int | str]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_trips": self._total_trips,
        }

    def _suggest_action(self, reason: BreakReason) -> str:
        """Suggest action based on break reason."""
        suggestions = {
            BreakReason.LOW_EVIDENCE: "Try a more specific query or add relevant keywords",
            BreakReason.NO_RELEVANT_RESULTS: "Reformulate query; this topic may not be covered",
            BreakReason.HIGH_UNCERTAINTY: "Provide more context or constraints in your query",
            BreakReason.CONFLICTING_SOURCES: "Ask about a specific aspect to resolve ambiguity",
            BreakReason.RATE_LIMIT: "Wait a moment before retrying",
            BreakReason.API_ERROR: "Retry the request or check service status",
        }
        return suggestions.get(reason, "Try reformulating your query")


class LowEvidenceHandler:
    """Handle low-evidence scenarios with graceful degradation."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def handle(
        self,
        retrieval_confidence: RetrievalConfidence,
        query: str,
    ) -> dict[str, str | bool]:
        """Handle a low-evidence scenario.

        Args:
            retrieval_confidence: Assessment of retrieval quality
            query: The original query

        Returns:
            Dict with handling result and user message
        """
        decision = self.circuit_breaker.check(retrieval_confidence=retrieval_confidence)

        if decision.allow:
            return {
                "proceed": True,
                "message": "",
                "degraded": False,
            }

        if decision.confidence_gap < 0.2:
            return {
                "proceed": True,
                "message": (
                    "Note: Limited evidence available for this query. "
                    "The response may be incomplete or less certain."
                ),
                "degraded": True,
            }

        return {
            "proceed": False,
            "message": (
                f"Unable to provide a reliable answer: {decision.message}. "
                f"Suggestion: {decision.suggested_action}"
            ),
            "degraded": False,
            "reason": decision.reason.value if decision.reason else "unknown",
        }
