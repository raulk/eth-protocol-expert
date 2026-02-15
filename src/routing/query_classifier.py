"""Query Classifier - Determine query complexity for routing (Phase 5)."""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import anthropic
import structlog

from ..config import DEFAULT_MODEL

logger = structlog.get_logger()


class QueryType(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


@dataclass
class ClassificationResult:
    """Result from query classification."""
    query: str
    query_type: QueryType
    confidence: float
    reasoning: str
    needs_decomposition: bool
    estimated_sub_questions: int


class QueryClassifier:
    """Classify queries as simple, complex, or multi-hop.

    Classification criteria:
    - Simple: Single entity lookup, direct question about one concept
      Examples: "What is EIP-4844?", "Who authored EIP-721?"

    - Complex: Multiple entities, comparison, evolution, timeline
      Examples: "Compare EIP-1559 and EIP-4844", "How did X evolve?"

    - Multi-hop: Requires reasoning across multiple retrievals
      Examples: "What EIPs depend on EIP-1559 and how do they relate to L2 scaling?"
    """

    COMPARISON_PATTERNS: ClassVar[list[str]] = [
        r'\bcompare\b',
        r'\bversus\b',
        r'\bvs\.?\b',
        r'\bdifference\s+between\b',
        r'\bhow\s+does\s+.+\s+differ\b',
        r'\bsimilarities\s+between\b',
        r'\bcontrast\b',
    ]

    EVOLUTION_PATTERNS: ClassVar[list[str]] = [
        r'\bevolve[d]?\b',
        r'\bhistory\s+of\b',
        r'\bover\s+time\b',
        r'\bprogression\b',
        r'\btimeline\b',
        r'\bchanged?\s+from\b',
        r'\bdevelop(ed|ment)?\b',
    ]

    MULTI_ENTITY_PATTERNS: ClassVar[list[str]] = [
        r'\band\b.*\band\b',
        r'\bmultiple\b',
        r'\bseveral\b',
        r'\ball\s+the\b',
        r'\beach\s+of\b',
    ]

    MULTI_HOP_PATTERNS: ClassVar[list[str]] = [
        r'\brelate\s+to\b',
        r'\bdepend(s|encies)?\s+on\b',
        r'\brequire[s]?\b.*\brequire[s]?\b',
        r'\bchain\s+of\b',
        r'\btransitive(ly)?\b',
        r'\bindirect(ly)?\b',
        r'\bimplications?\s+of\b',
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        use_llm: bool = True,
    ):
        """Initialize the query classifier.

        Args:
            api_key: Anthropic API key
            model: Model to use for LLM-based classification
            use_llm: Whether to use LLM for classification (more accurate but slower)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.use_llm = use_llm

        if self.use_llm and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

    async def classify(self, query: str) -> ClassificationResult:
        """Classify a query's complexity.

        Args:
            query: The user's question

        Returns:
            ClassificationResult with type, confidence, and reasoning
        """
        # First do rule-based classification
        rule_result = self._classify_rule_based(query)

        # If high confidence or LLM disabled, use rule-based result
        if rule_result.confidence >= 0.9 or not self.use_llm or not self.client:
            logger.debug(
                "query_classified_rule_based",
                query=query[:50],
                query_type=rule_result.query_type.value,
                confidence=rule_result.confidence,
            )
            return rule_result

        # For ambiguous cases, use LLM
        llm_result = await self._classify_llm(query)

        logger.debug(
            "query_classified_llm",
            query=query[:50],
            query_type=llm_result.query_type.value,
            confidence=llm_result.confidence,
        )

        return llm_result

    def classify_sync(self, query: str) -> ClassificationResult:
        """Synchronous rule-based classification."""
        return self._classify_rule_based(query)

    def _classify_rule_based(self, query: str) -> ClassificationResult:
        """Rule-based query classification."""
        query_lower = query.lower()

        # Count pattern matches
        comparison_count = self._count_pattern_matches(query_lower, self.COMPARISON_PATTERNS)
        evolution_count = self._count_pattern_matches(query_lower, self.EVOLUTION_PATTERNS)
        multi_entity_count = self._count_pattern_matches(query_lower, self.MULTI_ENTITY_PATTERNS)
        multi_hop_count = self._count_pattern_matches(query_lower, self.MULTI_HOP_PATTERNS)

        # Count EIP mentions
        eip_mentions = len(re.findall(r'eip-?\d+', query_lower))

        # Determine query type based on signals
        signals = {
            "comparison": comparison_count,
            "evolution": evolution_count,
            "multi_entity": multi_entity_count,
            "multi_hop": multi_hop_count,
            "eip_mentions": eip_mentions,
        }

        # Multi-hop: requires chaining retrievals
        if multi_hop_count >= 1 or (eip_mentions >= 2 and (comparison_count > 0 or evolution_count > 0)):
            query_type = QueryType.MULTI_HOP
            confidence = min(0.7 + (multi_hop_count * 0.1), 0.95)
            estimated_sub = max(2, eip_mentions + 1)
            reasoning = f"Multi-hop patterns detected: {signals}"

        # Complex: multiple entities or comparison/evolution
        elif comparison_count > 0 or evolution_count > 0 or eip_mentions >= 2:
            query_type = QueryType.COMPLEX
            confidence = min(0.6 + (comparison_count + evolution_count + eip_mentions) * 0.1, 0.9)
            estimated_sub = max(2, eip_mentions)
            reasoning = f"Complex query patterns: {signals}"

        # Simple: single entity lookup
        else:
            query_type = QueryType.SIMPLE
            confidence = 0.8 if eip_mentions <= 1 else 0.6
            estimated_sub = 1
            reasoning = "Simple query (single concept lookup)"

        needs_decomposition = query_type in (QueryType.COMPLEX, QueryType.MULTI_HOP)

        return ClassificationResult(
            query=query,
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            needs_decomposition=needs_decomposition,
            estimated_sub_questions=estimated_sub,
        )

    def _count_pattern_matches(self, text: str, patterns: list[str]) -> int:
        """Count how many patterns match in text."""
        count = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    async def _classify_llm(self, query: str) -> ClassificationResult:
        """LLM-based query classification for ambiguous cases."""
        prompt = self._build_classification_prompt(query)

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        return self._parse_llm_response(query, result_text)

    def _build_classification_prompt(self, query: str) -> str:
        """Build the classification prompt."""
        return f"""Classify this query about Ethereum protocol:

QUERY: {query}

Classify as one of:
- SIMPLE: Single entity lookup, direct question about one concept
  Examples: "What is EIP-4844?", "Who authored EIP-721?"

- COMPLEX: Multiple entities, comparison, evolution, timeline
  Examples: "Compare EIP-1559 and EIP-4844", "How did gas pricing evolve?"

- MULTI_HOP: Requires reasoning across multiple retrievals, dependencies, chains
  Examples: "What EIPs depend on EIP-1559 and how do they relate to L2 scaling?"

Return ONLY a JSON object:
{{
  "query_type": "simple" | "complex" | "multi_hop",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "estimated_sub_questions": number
}}"""

    def _parse_llm_response(self, query: str, response_text: str) -> ClassificationResult:
        """Parse LLM classification response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            query_type_str = parsed.get("query_type", "simple").lower()
            query_type = {
                "simple": QueryType.SIMPLE,
                "complex": QueryType.COMPLEX,
                "multi_hop": QueryType.MULTI_HOP,
            }.get(query_type_str, QueryType.SIMPLE)

            confidence = float(parsed.get("confidence", 0.7))
            reasoning = parsed.get("reasoning", "LLM classification")
            estimated_sub = int(parsed.get("estimated_sub_questions", 1))

            return ClassificationResult(
                query=query,
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                needs_decomposition=query_type in (QueryType.COMPLEX, QueryType.MULTI_HOP),
                estimated_sub_questions=estimated_sub,
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "failed_to_parse_classification",
                error=str(e),
                response=response_text[:200],
            )
            # Fallback to rule-based
            return self._classify_rule_based(query)
