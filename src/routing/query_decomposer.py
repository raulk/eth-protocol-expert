"""Query Decomposer - Break complex queries into sub-questions (Phase 5)."""

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum

import anthropic
import structlog

logger = structlog.get_logger()


class SynthesisStrategy(Enum):
    """Strategy for synthesizing sub-answers into final response."""
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


class QueryDecomposer:
    """Break complex queries into simpler sub-questions.

    Complex queries like "Compare the gas models of EIP-1559 and EIP-4844" need:
    1. Retrieve EIP-1559 gas model
    2. Retrieve EIP-4844 gas model
    3. Synthesize comparison

    The decomposer identifies sub-questions and the appropriate synthesis strategy.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose a complex query into sub-questions.

        Args:
            query: The complex question to decompose

        Returns:
            DecompositionResult with sub-questions and synthesis strategy
        """
        # First try rule-based decomposition for common patterns
        rule_result = self._decompose_rule_based(query)
        if rule_result.is_decomposed:
            logger.debug(
                "query_decomposed_rule_based",
                query=query[:50],
                num_sub_questions=len(rule_result.sub_questions),
                strategy=rule_result.synthesis_strategy.value,
            )
            return rule_result

        # Use LLM for complex decomposition
        llm_result = await self._decompose_llm(query)

        logger.info(
            "query_decomposed_llm",
            query=query[:50],
            num_sub_questions=len(llm_result.sub_questions),
            strategy=llm_result.synthesis_strategy.value,
        )

        return llm_result

    def decompose_sync(self, query: str) -> DecompositionResult:
        """Synchronous rule-based decomposition only."""
        return self._decompose_rule_based(query)

    def _decompose_rule_based(self, query: str) -> DecompositionResult:
        """Rule-based decomposition for common query patterns."""
        query_lower = query.lower()

        # Extract EIP mentions
        eip_pattern = re.compile(r'eip-?(\d+)', re.IGNORECASE)
        eip_mentions = eip_pattern.findall(query)
        eip_ids = [f"EIP-{num}" for num in eip_mentions]

        # Pattern: "Compare X and Y"
        compare_match = re.search(
            r'compare\s+(?:the\s+)?(.+?)\s+(?:and|with|to|vs\.?)\s+(.+?)(?:\s*\?|$)',
            query_lower,
            re.IGNORECASE,
        )
        if compare_match and len(eip_ids) >= 2:
            sub_questions = []
            for i, eip_id in enumerate(eip_ids[:2]):
                sub_questions.append(SubQuestion(
                    text=f"What is the gas model/mechanism of {eip_id}?",
                    index=i,
                    entity=eip_id,
                    focus="gas_model",
                ))
            return DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                synthesis_strategy=SynthesisStrategy.COMPARISON,
                reasoning=f"Comparison query detected for {eip_ids}",
                is_decomposed=True,
            )

        # Pattern: "How did X evolve from A to B"
        evolve_match = re.search(
            r'(?:how\s+did|evolution\s+of|history\s+of|timeline\s+of)\s+(.+?)(?:\s+evolve|\s+from|\s*\?|$)',
            query_lower,
            re.IGNORECASE,
        )
        if evolve_match and len(eip_ids) >= 2:
            sub_questions = []
            for i, eip_id in enumerate(eip_ids):
                sub_questions.append(SubQuestion(
                    text=f"What does {eip_id} propose and when was it created?",
                    index=i,
                    entity=eip_id,
                    focus="timeline",
                ))
            return DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                synthesis_strategy=SynthesisStrategy.TIMELINE,
                reasoning=f"Evolution/timeline query for {eip_ids}",
                is_decomposed=True,
            )

        # Pattern: Multiple EIPs without clear comparison structure
        if len(eip_ids) >= 2:
            sub_questions = []
            for i, eip_id in enumerate(eip_ids):
                sub_questions.append(SubQuestion(
                    text=f"What is {eip_id} and what does it propose?",
                    index=i,
                    entity=eip_id,
                    focus="overview",
                ))
            return DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                synthesis_strategy=SynthesisStrategy.AGGREGATION,
                reasoning=f"Multiple EIPs mentioned: {eip_ids}",
                is_decomposed=True,
            )

        # No decomposition needed
        return DecompositionResult(
            original_query=query,
            sub_questions=[SubQuestion(text=query, index=0)],
            synthesis_strategy=SynthesisStrategy.EXPLANATION,
            reasoning="Query does not require decomposition",
            is_decomposed=False,
        )

    async def _decompose_llm(self, query: str) -> DecompositionResult:
        """LLM-based query decomposition for complex cases."""
        prompt = self._build_decomposition_prompt(query)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        return self._parse_llm_response(query, result_text)

    def _build_decomposition_prompt(self, query: str) -> str:
        """Build the decomposition prompt."""
        return f"""Given this complex question about Ethereum protocol:
"{query}"

Break it into simpler sub-questions that can be answered independently.
Each sub-question should:
1. Target a single concept or entity
2. Be answerable with a single retrieval
3. Contribute to answering the original question

Also determine the synthesis strategy:
- comparison: Compare properties of multiple things
- timeline: Arrange information chronologically
- explanation: Explain a concept or mechanism
- aggregation: Combine related facts

Return ONLY a JSON object:
{{
  "sub_questions": [
    {{"text": "question text", "entity": "EIP-1234 or null", "focus": "aspect being asked about"}}
  ],
  "synthesis_strategy": "comparison" | "timeline" | "explanation" | "aggregation",
  "reasoning": "brief explanation of decomposition"
}}"""

    def _parse_llm_response(self, query: str, response_text: str) -> DecompositionResult:
        """Parse LLM decomposition response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            sub_questions = []
            for i, sq in enumerate(parsed.get("sub_questions", [])):
                if isinstance(sq, str):
                    sub_questions.append(SubQuestion(text=sq, index=i))
                else:
                    sub_questions.append(SubQuestion(
                        text=sq.get("text", ""),
                        index=i,
                        entity=sq.get("entity"),
                        focus=sq.get("focus"),
                    ))

            strategy_str = parsed.get("synthesis_strategy", "explanation").lower()
            strategy = {
                "comparison": SynthesisStrategy.COMPARISON,
                "timeline": SynthesisStrategy.TIMELINE,
                "explanation": SynthesisStrategy.EXPLANATION,
                "aggregation": SynthesisStrategy.AGGREGATION,
            }.get(strategy_str, SynthesisStrategy.EXPLANATION)

            reasoning = parsed.get("reasoning", "LLM decomposition")

            return DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                synthesis_strategy=strategy,
                reasoning=reasoning,
                is_decomposed=len(sub_questions) > 1,
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "failed_to_parse_decomposition",
                error=str(e),
                response=response_text[:200],
            )
            # Fallback: return original query
            return DecompositionResult(
                original_query=query,
                sub_questions=[SubQuestion(text=query, index=0)],
                synthesis_strategy=SynthesisStrategy.EXPLANATION,
                reasoning="Failed to parse LLM response, using original query",
                is_decomposed=False,
            )

    def estimate_sub_questions(self, query: str) -> int:
        """Estimate number of sub-questions without full decomposition.

        Useful for budget planning before decomposition.
        """
        eip_pattern = re.compile(r'eip-?\d+', re.IGNORECASE)
        eip_count = len(eip_pattern.findall(query))

        # Heuristic: at least as many sub-questions as EIP mentions
        base_count = max(1, eip_count)

        # Comparison adds a synthesis step
        if re.search(r'\b(compare|versus|vs\.?|difference)\b', query, re.IGNORECASE):
            return base_count + 1

        return base_count
