"""Budget Enforcer - Hard caps on agent retrieval operations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.agents.react_agent import AgentState

logger = structlog.get_logger()


@dataclass
class AgentBudget:
    """Budget constraints for agent operations.

    Hard caps prevent runaway agents from consuming excessive resources.
    """

    max_retrievals: int = 5
    max_tokens: int = 16000
    max_llm_calls: int = 10


@dataclass
class BudgetUsage:
    """Track current budget consumption."""

    retrievals: int = 0
    tokens: int = 0
    llm_calls: int = 0
    retrieval_history: list[int] = field(default_factory=list)


class BudgetEnforcer:
    """Enforce hard budget limits on agent operations.

    Tracks and limits:
    - Number of retrieval operations
    - Total tokens retrieved
    - Number of LLM calls made

    This prevents infinite loops and excessive resource consumption.
    """

    def __init__(self, budget: AgentBudget):
        self.budget = budget
        self.usage = BudgetUsage()

    def reset(self) -> None:
        """Reset usage tracking for a new query."""
        self.usage = BudgetUsage()

    def check_budget(self, state: "AgentState") -> bool:
        """Check if there's remaining budget for operations.

        Args:
            state: Current agent state

        Returns:
            True if budget remains, False if exhausted
        """
        has_retrieval_budget = self.usage.retrievals < self.budget.max_retrievals
        has_token_budget = self.usage.tokens < self.budget.max_tokens
        has_llm_budget = self.usage.llm_calls < self.budget.max_llm_calls

        if not has_retrieval_budget:
            logger.warning(
                "budget_exhausted_retrievals",
                used=self.usage.retrievals,
                max=self.budget.max_retrievals,
            )

        if not has_token_budget:
            logger.warning(
                "budget_exhausted_tokens",
                used=self.usage.tokens,
                max=self.budget.max_tokens,
            )

        if not has_llm_budget:
            logger.warning(
                "budget_exhausted_llm_calls",
                used=self.usage.llm_calls,
                max=self.budget.max_llm_calls,
            )

        return has_retrieval_budget and has_token_budget and has_llm_budget

    def can_retrieve(self) -> bool:
        """Check if a retrieval operation is allowed."""
        return self.usage.retrievals < self.budget.max_retrievals

    def can_call_llm(self) -> bool:
        """Check if an LLM call is allowed."""
        return self.usage.llm_calls < self.budget.max_llm_calls

    def deduct_retrieval(self, state: "AgentState", tokens: int) -> bool:
        """Record a retrieval operation.

        Args:
            state: Current agent state
            tokens: Number of tokens retrieved

        Returns:
            True if deduction succeeded, False if would exceed budget
        """
        if not self.can_retrieve():
            return False

        if self.usage.tokens + tokens > self.budget.max_tokens:
            logger.warning(
                "retrieval_would_exceed_token_budget",
                current_tokens=self.usage.tokens,
                requested_tokens=tokens,
                max_tokens=self.budget.max_tokens,
            )
            return False

        self.usage.retrievals += 1
        self.usage.tokens += tokens
        self.usage.retrieval_history.append(tokens)

        logger.debug(
            "budget_deducted_retrieval",
            retrieval_count=self.usage.retrievals,
            tokens_added=tokens,
            total_tokens=self.usage.tokens,
        )

        return True

    def deduct_llm_call(self, state: "AgentState") -> bool:
        """Record an LLM call.

        Args:
            state: Current agent state

        Returns:
            True if deduction succeeded, False if would exceed budget
        """
        if not self.can_call_llm():
            return False

        self.usage.llm_calls += 1

        logger.debug(
            "budget_deducted_llm_call",
            llm_call_count=self.usage.llm_calls,
        )

        return True

    def get_remaining(self) -> dict[str, int]:
        """Get remaining budget for all resource types.

        Returns:
            Dict with remaining counts for each resource type
        """
        return {
            "retrievals": self.budget.max_retrievals - self.usage.retrievals,
            "tokens": self.budget.max_tokens - self.usage.tokens,
            "llm_calls": self.budget.max_llm_calls - self.usage.llm_calls,
        }

    def get_utilization(self) -> dict[str, float]:
        """Get utilization percentage for each resource type."""
        return {
            "retrievals": self.usage.retrievals / self.budget.max_retrievals,
            "tokens": self.usage.tokens / self.budget.max_tokens,
            "llm_calls": self.usage.llm_calls / self.budget.max_llm_calls,
        }

    def get_summary(self) -> dict:
        """Get complete budget summary."""
        return {
            "budget": {
                "max_retrievals": self.budget.max_retrievals,
                "max_tokens": self.budget.max_tokens,
                "max_llm_calls": self.budget.max_llm_calls,
            },
            "usage": {
                "retrievals": self.usage.retrievals,
                "tokens": self.usage.tokens,
                "llm_calls": self.usage.llm_calls,
            },
            "remaining": self.get_remaining(),
            "utilization": self.get_utilization(),
        }
