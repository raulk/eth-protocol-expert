"""Query Analyzer - Analyze query complexity for adaptive budget selection."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import structlog

from src.agents.budget_enforcer import AgentBudget

logger = structlog.get_logger()


class QueryComplexity(Enum):
    """Complexity level of a query."""

    SIMPLE = "simple"  # Single-hop factual lookup
    MODERATE = "moderate"  # Requires synthesis from 2-3 sources
    COMPLEX = "complex"  # Multi-hop reasoning, comparisons
    RESEARCH = "research"  # Deep dive, timeline, comprehensive analysis


@dataclass
class QueryAnalysis:
    """Analysis result for a query."""

    query: str
    complexity: QueryComplexity
    signals: list[str]
    suggested_budget: AgentBudget
    suggested_mode: str  # simple, hybrid, graph, agentic
    multi_hop: bool
    comparison: bool
    timeline: bool
    requires_graph: bool


class QueryAnalyzer:
    """Analyze queries to determine complexity and resource requirements.

    Uses pattern matching and heuristics to classify queries and
    suggest appropriate retrieval strategies and budgets.
    """

    # Patterns indicating multi-hop reasoning
    MULTI_HOP_PATTERNS: ClassVar[list[str]] = [
        r"(?:how|what)\s+.*\s+(?:affect|impact|influence|relate|depend)",
        r"(?:compare|contrast|difference|versus|vs\.?)",
        r"(?:evolution|history|timeline|progression)",
        r"(?:why|explain|reason|rationale)\s+.*\s+(?:chose|decided|proposed)",
        r"(?:pros\s+and\s+cons|trade-?offs?|advantages\s+and\s+disadvantages)",
    ]

    # Patterns indicating comparison queries
    COMPARISON_PATTERNS: ClassVar[list[str]] = [
        r"(?:compare|contrast|difference|versus|vs\.?)",
        r"(?:better|worse|preferred|alternative)",
        r"(?:advantages?\s+of\s+\w+\s+over)",
        r"eip-?\d+\s+(?:and|vs\.?|versus)\s+eip-?\d+",
    ]

    # Patterns indicating timeline/history queries
    TIMELINE_PATTERNS: ClassVar[list[str]] = [
        r"(?:when|timeline|history|evolution|progression)",
        r"(?:first|earliest|original|initial)",
        r"(?:changed|evolved|developed|progressed)",
        r"(?:before|after|since|until)",
    ]

    # Patterns indicating graph-augmented retrieval
    GRAPH_PATTERNS: ClassVar[list[str]] = [
        r"(?:depends?\s+on|requires|needs)",
        r"(?:dependent|dependency|dependencies)",
        r"(?:related|relationship|connected)",
        r"(?:supersedes?|replaced|obsolete)",
        r"(?:eips?\s+that\s+(?:use|implement|require|depend))",
    ]

    # Simple factual patterns
    SIMPLE_PATTERNS: ClassVar[list[str]] = [
        r"^what\s+is\s+",
        r"^define\s+",
        r"^explain\s+(?:the\s+)?(?:concept|term|meaning)",
        r"^(?:gas\s+cost|fee|price)\s+of",
    ]

    # Budget presets for each complexity level
    BUDGET_PRESETS: ClassVar[dict[QueryComplexity, AgentBudget]] = {
        QueryComplexity.SIMPLE: AgentBudget(max_retrievals=2, max_tokens=4000, max_llm_calls=3),
        QueryComplexity.MODERATE: AgentBudget(max_retrievals=4, max_tokens=8000, max_llm_calls=6),
        QueryComplexity.COMPLEX: AgentBudget(max_retrievals=6, max_tokens=12000, max_llm_calls=10),
        QueryComplexity.RESEARCH: AgentBudget(
            max_retrievals=10, max_tokens=20000, max_llm_calls=15
        ),
    }

    def __init__(self) -> None:
        self.multi_hop_patterns = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        self.comparison_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self.timeline_patterns = [re.compile(p, re.IGNORECASE) for p in self.TIMELINE_PATTERNS]
        self.graph_patterns = [re.compile(p, re.IGNORECASE) for p in self.GRAPH_PATTERNS]
        self.simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and determine its complexity.

        Args:
            query: The user's query

        Returns:
            QueryAnalysis with complexity, signals, and suggested budget
        """
        signals = []

        # Check for simple patterns first
        is_simple = any(p.search(query) for p in self.simple_patterns)
        if is_simple:
            signals.append("simple_pattern_match")

        # Check for multi-hop indicators
        multi_hop = any(p.search(query) for p in self.multi_hop_patterns)
        if multi_hop:
            signals.append("multi_hop_reasoning")

        # Check for comparison
        comparison = any(p.search(query) for p in self.comparison_patterns)
        if comparison:
            signals.append("comparison_query")

        # Check for timeline/history
        timeline = any(p.search(query) for p in self.timeline_patterns)
        if timeline:
            signals.append("timeline_query")

        # Check for graph traversal
        requires_graph = any(p.search(query) for p in self.graph_patterns)
        if requires_graph:
            signals.append("graph_traversal_needed")

        # Count EIP references
        eip_refs = re.findall(r"eip-?\d+", query, re.IGNORECASE)
        if len(eip_refs) > 1:
            signals.append(f"multiple_eip_refs_{len(eip_refs)}")

        # Determine complexity
        complexity = self._determine_complexity(
            is_simple=is_simple,
            multi_hop=multi_hop,
            comparison=comparison,
            timeline=timeline,
            requires_graph=requires_graph,
            eip_count=len(eip_refs),
            query_length=len(query.split()),
        )

        # Determine suggested mode
        suggested_mode = self._suggest_mode(
            complexity=complexity,
            requires_graph=requires_graph,
            comparison=comparison,
            timeline=timeline,
        )

        logger.debug(
            "query_analyzed",
            query=query[:50],
            complexity=complexity.value,
            signals=signals,
            suggested_mode=suggested_mode,
        )

        return QueryAnalysis(
            query=query,
            complexity=complexity,
            signals=signals,
            suggested_budget=self.BUDGET_PRESETS[complexity],
            suggested_mode=suggested_mode,
            multi_hop=multi_hop,
            comparison=comparison,
            timeline=timeline,
            requires_graph=requires_graph,
        )

    def _determine_complexity(
        self,
        is_simple: bool,
        multi_hop: bool,
        comparison: bool,
        timeline: bool,
        requires_graph: bool,
        eip_count: int,
        query_length: int,
    ) -> QueryComplexity:
        """Determine query complexity based on signals."""
        # Calculate complexity score
        score = 0

        if is_simple and not (multi_hop or comparison or timeline):
            return QueryComplexity.SIMPLE

        if multi_hop:
            score += 2
        if comparison:
            score += 2
        if timeline:
            score += 2
        if requires_graph:
            score += 1
        if eip_count > 1:
            score += eip_count - 1
        if query_length > 20:
            score += 1

        if score >= 6:
            return QueryComplexity.RESEARCH
        elif score >= 4:
            return QueryComplexity.COMPLEX
        elif score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _suggest_mode(
        self,
        complexity: QueryComplexity,
        requires_graph: bool,
        comparison: bool,
        timeline: bool,
    ) -> str:
        """Suggest retrieval mode based on query characteristics."""
        if complexity == QueryComplexity.SIMPLE:
            return "simple"

        if requires_graph:
            return "graph"

        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]:
            return "agentic"

        if comparison or timeline:
            return "hybrid"

        return "hybrid"

    def get_budget_for_complexity(self, complexity: QueryComplexity) -> AgentBudget:
        """Get budget preset for a complexity level."""
        return self.BUDGET_PRESETS[complexity]
