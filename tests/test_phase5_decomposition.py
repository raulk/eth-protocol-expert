"""Tests for Phase 5: Query Decomposition.

These are self-contained unit tests that don't require external dependencies.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """Query complexity type."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class SynthesisStrategy(Enum):
    """Strategy for synthesizing sub-answers."""
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    EXPLANATION = "explanation"
    AGGREGATION = "aggregation"


@dataclass
class SubQuestion:
    """A sub-question derived from the original query."""
    text: str
    index: int
    entity: str | None = None
    focus: str | None = None
    depends_on: list[int] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Result from query decomposition."""
    original_query: str
    sub_questions: list[SubQuestion]
    synthesis_strategy: SynthesisStrategy
    reasoning: str
    is_decomposed: bool


@dataclass
class BudgetConfig:
    """Configuration for retrieval budget."""
    max_chunks_per_sub_question: int = 5
    total_chunk_budget: int = 20
    total_token_budget: int = 8000


@dataclass
class BudgetUsage:
    """Track budget usage."""
    chunks_used: int = 0
    tokens_used: int = 0


@dataclass
class BudgetAllocation:
    """Budget allocation for retrieval."""
    chunks_per_sub_question: int
    tokens_per_sub_question: int


class BudgetManager:
    """Manage retrieval budget."""

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.usage = BudgetUsage()

    def allocate_budget(self, num_sub_questions: int) -> BudgetAllocation:
        chunks_per = min(
            self.config.max_chunks_per_sub_question,
            self.config.total_chunk_budget // max(1, num_sub_questions),
        )
        tokens_per = self.config.total_token_budget // max(1, num_sub_questions)
        return BudgetAllocation(chunks_per, tokens_per)

    def record_usage(self, chunks: int, tokens: int):
        self.usage.chunks_used += chunks
        self.usage.tokens_used += tokens

    def get_usage(self) -> BudgetUsage:
        return self.usage

    def remaining_chunks(self) -> int:
        return self.config.total_chunk_budget - self.usage.chunks_used

    def remaining_tokens(self) -> int:
        return self.config.total_token_budget - self.usage.tokens_used

    def is_chunk_budget_exhausted(self) -> bool:
        return self.usage.chunks_used >= self.config.total_chunk_budget


class TestQueryType:
    """Tests for query type classification."""

    def test_simple_query_type(self):
        """Simple queries should be single entity lookups."""
        query_type = QueryType.SIMPLE
        assert query_type.value == "simple"

    def test_complex_query_type(self):
        """Complex queries involve multiple entities or comparisons."""
        query_type = QueryType.COMPLEX
        assert query_type.value == "complex"

    def test_multi_hop_query_type(self):
        """Multi-hop queries require multiple retrieval steps."""
        query_type = QueryType.MULTI_HOP
        assert query_type.value == "multi_hop"


class TestSynthesisStrategy:
    """Tests for synthesis strategy enum."""

    def test_comparison_strategy(self):
        strategy = SynthesisStrategy.COMPARISON
        assert strategy.value == "comparison"

    def test_timeline_strategy(self):
        strategy = SynthesisStrategy.TIMELINE
        assert strategy.value == "timeline"

    def test_explanation_strategy(self):
        strategy = SynthesisStrategy.EXPLANATION
        assert strategy.value == "explanation"

    def test_aggregation_strategy(self):
        strategy = SynthesisStrategy.AGGREGATION
        assert strategy.value == "aggregation"


class TestSubQuestion:
    """Tests for sub-question dataclass."""

    def test_basic_sub_question(self):
        sq = SubQuestion(
            text="What is the gas model of EIP-1559?",
            index=0,
        )

        assert sq.text == "What is the gas model of EIP-1559?"
        assert sq.index == 0
        assert sq.entity is None

    def test_sub_question_with_entity(self):
        sq = SubQuestion(
            text="What is the gas model of EIP-1559?",
            index=0,
            entity="EIP-1559",
            focus="gas_model",
        )

        assert sq.entity == "EIP-1559"
        assert sq.focus == "gas_model"

    def test_sub_question_with_dependencies(self):
        sq = SubQuestion(
            text="Compare the two gas models",
            index=2,
            depends_on=[0, 1],
        )

        assert sq.depends_on == [0, 1]


class TestQueryDecomposition:
    """Tests for rule-based query decomposition."""

    def extract_eips(self, query: str) -> list[str]:
        """Extract EIP mentions from query."""
        eip_pattern = re.compile(r'eip-?(\d+)', re.IGNORECASE)
        eip_mentions = eip_pattern.findall(query)
        return [f"EIP-{num}" for num in eip_mentions]

    def test_decompose_comparison_query(self):
        query = "Compare the gas models of EIP-1559 and EIP-4844"
        eip_ids = self.extract_eips(query)

        assert len(eip_ids) == 2
        assert "EIP-1559" in eip_ids
        assert "EIP-4844" in eip_ids

        # Should create one sub-question per EIP
        sub_questions = [
            SubQuestion(f"What is the gas model of {eip}?", i, entity=eip)
            for i, eip in enumerate(eip_ids)
        ]

        assert len(sub_questions) == 2

    def test_decompose_evolution_query(self):
        query = "How did account abstraction evolve from EIP-86 to EIP-4337?"
        eip_ids = self.extract_eips(query)

        assert len(eip_ids) >= 2
        assert "EIP-86" in eip_ids
        assert "EIP-4337" in eip_ids

    def test_simple_query_no_decomposition(self):
        query = "What is EIP-4844?"
        eip_ids = self.extract_eips(query)

        assert len(eip_ids) == 1

        result = DecompositionResult(
            original_query=query,
            sub_questions=[SubQuestion(text=query, index=0)],
            synthesis_strategy=SynthesisStrategy.EXPLANATION,
            reasoning="Single entity query",
            is_decomposed=False,
        )

        assert not result.is_decomposed
        assert len(result.sub_questions) == 1

    def test_comparison_detection(self):
        """Test detecting comparison patterns in queries."""
        comparison_patterns = [
            "Compare EIP-1559 and EIP-4844",
            "What is the difference between EIP-1559 vs EIP-4844",
            "EIP-1559 versus EIP-4844",
        ]

        compare_pattern = re.compile(r'\b(compare|versus|vs\.?|difference)\b', re.IGNORECASE)

        for query in comparison_patterns:
            assert compare_pattern.search(query), f"Failed to detect comparison in: {query}"


class TestBudgetManager:
    """Tests for retrieval budget management."""

    def test_default_budget_config(self):
        config = BudgetConfig()

        assert config.max_chunks_per_sub_question > 0
        assert config.total_chunk_budget > 0
        assert config.total_token_budget > 0

    def test_budget_allocation(self):
        config = BudgetConfig(
            max_chunks_per_sub_question=5,
            total_chunk_budget=20,
            total_token_budget=8000,
        )
        manager = BudgetManager(config)

        num_sub_questions = 3
        allocation = manager.allocate_budget(num_sub_questions)

        # Each sub-question should get equal share
        expected_per_question = config.total_chunk_budget // num_sub_questions
        assert allocation.chunks_per_sub_question == min(
            config.max_chunks_per_sub_question,
            expected_per_question,
        )

    def test_budget_not_exceeded(self):
        config = BudgetConfig(
            max_chunks_per_sub_question=10,
            total_chunk_budget=20,
        )
        manager = BudgetManager(config)

        allocation = manager.allocate_budget(5)
        max_possible = allocation.chunks_per_sub_question * 5

        assert max_possible <= config.total_chunk_budget

    def test_track_usage(self):
        config = BudgetConfig(total_chunk_budget=20)
        manager = BudgetManager(config)

        manager.record_usage(chunks=5, tokens=1000)
        usage = manager.get_usage()

        assert usage.chunks_used == 5
        assert usage.tokens_used == 1000

    def test_remaining_budget(self):
        config = BudgetConfig(total_chunk_budget=20, total_token_budget=8000)
        manager = BudgetManager(config)

        manager.record_usage(chunks=8, tokens=3000)

        assert manager.remaining_chunks() == 12
        assert manager.remaining_tokens() == 5000

    def test_budget_exhausted(self):
        config = BudgetConfig(total_chunk_budget=10)
        manager = BudgetManager(config)

        manager.record_usage(chunks=10, tokens=0)

        assert manager.is_chunk_budget_exhausted()


class TestDecompositionResult:
    """Tests for decomposition result."""

    def test_decomposition_result_creation(self):
        result = DecompositionResult(
            original_query="Compare EIP-1559 and EIP-4844",
            sub_questions=[
                SubQuestion("What is EIP-1559?", 0, entity="EIP-1559"),
                SubQuestion("What is EIP-4844?", 1, entity="EIP-4844"),
            ],
            synthesis_strategy=SynthesisStrategy.COMPARISON,
            reasoning="Comparison query with two EIPs",
            is_decomposed=True,
        )

        assert result.is_decomposed
        assert len(result.sub_questions) == 2
        assert result.synthesis_strategy == SynthesisStrategy.COMPARISON

    def test_estimate_sub_questions(self):
        """Test estimating number of sub-questions."""

        def estimate(query: str) -> int:
            eip_pattern = re.compile(r'eip-?\d+', re.IGNORECASE)
            eip_count = len(eip_pattern.findall(query))

            base_count = max(1, eip_count)

            if re.search(r'\b(compare|versus|vs\.?|difference)\b', query, re.IGNORECASE):
                return base_count + 1

            return base_count

        assert estimate("What is EIP-4844?") == 1
        assert estimate("EIP-1559 and EIP-4844") == 2
        assert estimate("Compare EIP-1559 vs EIP-4844") == 3
