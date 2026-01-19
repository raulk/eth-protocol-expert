"""Backtrack - Detect and abandon dead-end retrieval paths."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.agents.react_agent import AgentState

logger = structlog.get_logger()


@dataclass
class BacktrackDecision:
    """Decision about whether to backtrack from current path."""

    should_backtrack: bool
    reason: str
    alternative_query: str | None = None
    suggested_action: str | None = None


@dataclass
class RetrievalAttempt:
    """Record of a retrieval attempt for cycle detection."""

    query: str
    mode: str
    num_results: int
    was_useful: bool


class Backtracker:
    """Detect dead-end retrieval paths and suggest alternatives.

    Monitors for:
    - Repeated retrieval failures (low-quality results)
    - Circular retrieval patterns (same queries repeated)
    - Budget exhaustion approaching
    - Diminishing returns (each retrieval adds less value)
    """

    def __init__(
        self,
        max_failed_attempts: int = 3,
        similarity_threshold: float = 0.8,
        min_useful_results: int = 1,
    ):
        self.max_failed_attempts = max_failed_attempts
        self.similarity_threshold = similarity_threshold
        self.min_useful_results = min_useful_results
        self.attempts: list[RetrievalAttempt] = []
        self.query_history: list[str] = []

    def reset(self) -> None:
        """Reset tracking for a new query session."""
        self.attempts = []
        self.query_history = []

    def record_attempt(
        self,
        query: str,
        mode: str,
        num_results: int,
        was_useful: bool,
    ) -> None:
        """Record a retrieval attempt for pattern detection.

        Args:
            query: The query used
            mode: Retrieval mode used
            num_results: Number of results returned
            was_useful: Whether results were deemed useful
        """
        self.attempts.append(
            RetrievalAttempt(
                query=query,
                mode=mode,
                num_results=num_results,
                was_useful=was_useful,
            )
        )
        self.query_history.append(query.lower().strip())

        logger.debug(
            "recorded_retrieval_attempt",
            query=query[:50],
            mode=mode,
            num_results=num_results,
            was_useful=was_useful,
            total_attempts=len(self.attempts),
        )

    async def check(self, state: "AgentState") -> BacktrackDecision:
        """Check if current retrieval path should be abandoned.

        Args:
            state: Current agent state

        Returns:
            BacktrackDecision with recommendation
        """
        if self._detect_repeated_failures():
            return BacktrackDecision(
                should_backtrack=True,
                reason="Multiple consecutive retrieval failures",
                alternative_query=self._suggest_alternative_query(state.query),
                suggested_action="Try a different query formulation or retrieval mode",
            )

        circular = self._detect_circular_retrieval()
        if circular:
            return BacktrackDecision(
                should_backtrack=True,
                reason=f"Circular retrieval detected: repeating '{circular}'",
                alternative_query=self._generate_different_query(state.query),
                suggested_action="Break the cycle with a fundamentally different approach",
            )

        if self._detect_diminishing_returns():
            return BacktrackDecision(
                should_backtrack=True,
                reason="Diminishing returns from additional retrievals",
                suggested_action="Synthesize answer from current information",
            )

        if state.budget_remaining <= 1:
            return BacktrackDecision(
                should_backtrack=True,
                reason="Budget nearly exhausted",
                suggested_action="Generate best answer with available information",
            )

        return BacktrackDecision(
            should_backtrack=False,
            reason="No issues detected, continue retrieval",
        )

    def _detect_repeated_failures(self) -> bool:
        """Detect consecutive failed retrieval attempts."""
        if len(self.attempts) < self.max_failed_attempts:
            return False

        recent = self.attempts[-self.max_failed_attempts :]
        return all(not a.was_useful for a in recent)

    def _detect_circular_retrieval(self) -> str | None:
        """Detect if the same query is being repeated.

        Returns:
            The repeated query if circular pattern detected, None otherwise
        """
        if len(self.query_history) < 2:
            return None

        current = self.query_history[-1]

        for i, prev in enumerate(self.query_history[:-1]):
            if self._query_similarity(current, prev) > self.similarity_threshold:
                logger.debug(
                    "circular_query_detected",
                    current=current[:50],
                    previous=prev[:50],
                    distance=len(self.query_history) - i - 1,
                )
                return prev

        return None

    def _query_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between two queries."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _detect_diminishing_returns(self) -> bool:
        """Detect if recent retrievals are adding less value."""
        if len(self.attempts) < 3:
            return False

        recent = self.attempts[-3:]
        useful_count = sum(1 for a in recent if a.was_useful)

        if useful_count == 0:
            return True

        avg_results = sum(a.num_results for a in recent) / len(recent)
        return avg_results < self.min_useful_results

    def _suggest_alternative_query(self, original_query: str) -> str:
        """Suggest an alternative query based on the original."""
        words = original_query.split()

        if len(words) > 5:
            return " ".join(words[:5])

        if any(word.lower() in ["how", "what", "why", "when"] for word in words):
            return f"explain {' '.join(words[1:])}"

        return f"details about {original_query}"

    def _generate_different_query(self, original_query: str) -> str:
        """Generate a fundamentally different query approach."""
        if "eip" in original_query.lower():
            return original_query.lower().replace("eip-", "EIP ").replace("eip ", "")

        if len(original_query.split()) > 3:
            return " ".join(original_query.split()[:3])

        return f"overview of {original_query}"

    def get_attempt_summary(self) -> dict:
        """Get summary of retrieval attempts."""
        if not self.attempts:
            return {
                "total_attempts": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
            }

        successful = sum(1 for a in self.attempts if a.was_useful)
        failed = len(self.attempts) - successful

        return {
            "total_attempts": len(self.attempts),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.attempts),
            "unique_queries": len(set(self.query_history)),
        }
