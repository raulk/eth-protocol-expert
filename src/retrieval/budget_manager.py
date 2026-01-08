"""Budget Manager - Track and enforce retrieval limits (Phase 5)."""

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class BudgetConfig:
    """Configuration for retrieval budget limits."""
    max_chunks_per_sub_question: int = 5
    total_max_chunks: int = 20
    total_max_tokens: int = 8000
    reserve_tokens_for_synthesis: int = 2000


@dataclass
class BudgetUsage:
    """Current budget usage tracking."""
    chunks_used: int = 0
    tokens_used: int = 0
    chunks_per_sub_question: dict[int, int] = field(default_factory=dict)
    tokens_per_sub_question: dict[int, int] = field(default_factory=dict)


@dataclass
class BudgetAllocation:
    """Allocation of budget for a sub-question."""
    sub_question_index: int
    max_chunks: int
    max_tokens: int
    is_limited: bool
    reason: str | None = None


class BudgetManager:
    """Manage retrieval budget across sub-questions.

    Enforces:
    - Max chunks per sub-question: 5
    - Total budget: 20 chunks
    - Token budget: 8000 tokens max

    The manager tracks usage and provides allocations to ensure
    fair distribution across sub-questions while staying within limits.
    """

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.usage = BudgetUsage()
        self._num_sub_questions = 0

    def reset(self) -> None:
        """Reset budget usage for a new query."""
        self.usage = BudgetUsage()
        self._num_sub_questions = 0

    def set_sub_question_count(self, count: int) -> None:
        """Set the expected number of sub-questions for budget planning.

        Args:
            count: Number of sub-questions to plan for
        """
        self._num_sub_questions = count
        logger.debug(
            "budget_planned",
            sub_questions=count,
            chunks_per_sq=self._calculate_fair_chunks_per_sq(),
            tokens_per_sq=self._calculate_fair_tokens_per_sq(),
        )

    def get_allocation(self, sub_question_index: int) -> BudgetAllocation:
        """Get budget allocation for a sub-question.

        Args:
            sub_question_index: Index of the sub-question (0-based)

        Returns:
            BudgetAllocation with limits for this sub-question
        """
        remaining_chunks = self.get_remaining_chunks()
        remaining_tokens = self.get_remaining_tokens()

        if remaining_chunks <= 0:
            return BudgetAllocation(
                sub_question_index=sub_question_index,
                max_chunks=0,
                max_tokens=0,
                is_limited=True,
                reason="Chunk budget exhausted",
            )

        if remaining_tokens <= 0:
            return BudgetAllocation(
                sub_question_index=sub_question_index,
                max_chunks=0,
                max_tokens=0,
                is_limited=True,
                reason="Token budget exhausted",
            )

        # Calculate fair share for remaining sub-questions
        remaining_sqs = max(1, self._num_sub_questions - sub_question_index)
        fair_chunks = min(
            self.config.max_chunks_per_sub_question,
            remaining_chunks // remaining_sqs,
        )
        fair_tokens = remaining_tokens // remaining_sqs

        max_chunks = max(1, fair_chunks)
        max_tokens = max(500, fair_tokens)

        is_limited = (
            max_chunks < self.config.max_chunks_per_sub_question or
            remaining_chunks < self.config.total_max_chunks // 2
        )

        return BudgetAllocation(
            sub_question_index=sub_question_index,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            is_limited=is_limited,
            reason="Budget constraints applied" if is_limited else None,
        )

    def record_usage(
        self,
        sub_question_index: int,
        chunks: int,
        tokens: int,
    ) -> bool:
        """Record budget usage for a sub-question retrieval.

        Args:
            sub_question_index: Index of the sub-question
            chunks: Number of chunks retrieved
            tokens: Number of tokens in retrieved chunks

        Returns:
            True if usage was recorded, False if it would exceed budget
        """
        # Check if this would exceed budget
        if self.usage.chunks_used + chunks > self.config.total_max_chunks:
            logger.warning(
                "budget_exceeded_chunks",
                requested=chunks,
                used=self.usage.chunks_used,
                max=self.config.total_max_chunks,
            )
            return False

        if self.usage.tokens_used + tokens > self.config.total_max_tokens:
            logger.warning(
                "budget_exceeded_tokens",
                requested=tokens,
                used=self.usage.tokens_used,
                max=self.config.total_max_tokens,
            )
            return False

        # Record usage
        self.usage.chunks_used += chunks
        self.usage.tokens_used += tokens
        self.usage.chunks_per_sub_question[sub_question_index] = chunks
        self.usage.tokens_per_sub_question[sub_question_index] = tokens

        logger.debug(
            "budget_usage_recorded",
            sub_question=sub_question_index,
            chunks=chunks,
            tokens=tokens,
            total_chunks=self.usage.chunks_used,
            total_tokens=self.usage.tokens_used,
        )

        return True

    def get_remaining_chunks(self) -> int:
        """Get number of remaining chunks in budget."""
        return self.config.total_max_chunks - self.usage.chunks_used

    def get_remaining_tokens(self) -> int:
        """Get number of remaining tokens in budget."""
        return self.config.total_max_tokens - self.usage.tokens_used

    def get_synthesis_token_budget(self) -> int:
        """Get remaining tokens available for synthesis.

        Returns tokens remaining after reserving for synthesis overhead.
        """
        return max(0, self.get_remaining_tokens() - self.config.reserve_tokens_for_synthesis)

    def is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return (
            self.usage.chunks_used >= self.config.total_max_chunks or
            self.usage.tokens_used >= self.config.total_max_tokens
        )

    def get_utilization_summary(self) -> dict:
        """Get summary of budget utilization."""
        return {
            "chunks_used": self.usage.chunks_used,
            "chunks_max": self.config.total_max_chunks,
            "chunks_utilization": self.usage.chunks_used / self.config.total_max_chunks,
            "tokens_used": self.usage.tokens_used,
            "tokens_max": self.config.total_max_tokens,
            "tokens_utilization": self.usage.tokens_used / self.config.total_max_tokens,
            "sub_questions_served": len(self.usage.chunks_per_sub_question),
        }

    def _calculate_fair_chunks_per_sq(self) -> int:
        """Calculate fair chunk allocation per sub-question."""
        if self._num_sub_questions <= 0:
            return self.config.max_chunks_per_sub_question

        fair = self.config.total_max_chunks // self._num_sub_questions
        return min(fair, self.config.max_chunks_per_sub_question)

    def _calculate_fair_tokens_per_sq(self) -> int:
        """Calculate fair token allocation per sub-question."""
        if self._num_sub_questions <= 0:
            return self.config.total_max_tokens

        # Reserve tokens for synthesis
        available = self.config.total_max_tokens - self.config.reserve_tokens_for_synthesis
        return available // self._num_sub_questions


class AdaptiveBudgetManager(BudgetManager):
    """Budget manager with adaptive allocation based on retrieval quality.

    Extends BudgetManager to allocate more budget to sub-questions
    with lower-quality initial retrievals.
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        quality_threshold: float = 0.7,
    ):
        super().__init__(config)
        self.quality_threshold = quality_threshold
        self._quality_scores: dict[int, float] = {}

    def record_quality(self, sub_question_index: int, similarity_score: float) -> None:
        """Record the quality of retrieval for a sub-question.

        Args:
            sub_question_index: Index of the sub-question
            similarity_score: Average similarity score of retrieved chunks
        """
        self._quality_scores[sub_question_index] = similarity_score

    def should_expand_retrieval(self, sub_question_index: int) -> bool:
        """Check if retrieval should be expanded due to low quality.

        Returns True if the retrieval quality was below threshold
        and there's remaining budget.
        """
        if sub_question_index not in self._quality_scores:
            return False

        quality = self._quality_scores[sub_question_index]
        has_budget = self.get_remaining_chunks() > 0

        return quality < self.quality_threshold and has_budget

    def get_expanded_allocation(self, sub_question_index: int) -> BudgetAllocation:
        """Get expanded allocation for low-quality retrieval.

        Allocates additional chunks if quality was below threshold.
        """
        base = self.get_allocation(sub_question_index)

        if not self.should_expand_retrieval(sub_question_index):
            return base

        # Allow up to 50% more chunks for low-quality retrievals
        expanded_chunks = min(
            base.max_chunks + base.max_chunks // 2,
            self.get_remaining_chunks(),
        )

        return BudgetAllocation(
            sub_question_index=sub_question_index,
            max_chunks=expanded_chunks,
            max_tokens=base.max_tokens,
            is_limited=base.is_limited,
            reason="Expanded allocation for low-quality retrieval",
        )
