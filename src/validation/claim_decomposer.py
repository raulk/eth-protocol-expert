"""Claim Decomposer - Break complex claims into atomic facts (Phase 2)."""

import asyncio
import json
import os
import re
from typing import ClassVar

import anthropic
import structlog

from ..config import DEFAULT_MODEL
from .nli_validator import AtomicFact

logger = structlog.get_logger()


class ClaimDecomposer:
    """Decompose complex claims into atomic, independently verifiable facts.

    Complex claims like "EIP-4844 introduces blob transactions that reduce L2
    costs by 10-100x and were proposed by Vitalik" contain multiple facts:
    1. EIP-4844 introduces blob transactions
    2. Blob transactions reduce L2 costs
    3. The cost reduction is 10-100x
    4. Vitalik proposed this

    Each atomic fact can be independently verified against evidence.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def decompose(self, claim_text: str) -> list[AtomicFact]:
        """Decompose a claim into atomic facts.

        Args:
            claim_text: The complex claim to decompose

        Returns:
            List of atomic facts extracted from the claim
        """
        prompt = self._build_prompt(claim_text)

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Parse the JSON response
        facts = self._parse_response(result_text, claim_text)

        logger.debug(
            "decomposed_claim",
            claim=claim_text[:50],
            num_facts=len(facts),
        )

        return facts

    def decompose_sync(self, claim_text: str) -> list[AtomicFact]:
        """Synchronous version of decompose."""
        prompt = self._build_prompt(claim_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        return self._parse_response(result_text, claim_text)

    def _build_prompt(self, claim_text: str) -> str:
        """Build the decomposition prompt."""
        return f"""Decompose this claim into atomic, independently verifiable facts.

Each atomic fact should:
1. Be a single assertion that can be true or false
2. Not depend on other facts to be understood
3. Be specific and precise
4. Not contain conjunctions (and, or, but)

CLAIM: {claim_text}

Return ONLY a JSON object in this exact format:
{{"facts": ["fact1", "fact2", "fact3"]}}

If the claim is already atomic (a single simple fact), return just that one fact.
Do not include any other text before or after the JSON."""

    def _parse_response(self, response_text: str, source_claim: str) -> list[AtomicFact]:
        """Parse the LLM response into atomic facts."""
        # Try to extract JSON from the response
        try:
            # Find JSON in response (handle potential markdown formatting)
            json_match = re.search(r'\{[^{}]*"facts"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            facts = parsed.get("facts", [])

            return [
                AtomicFact(text=fact.strip(), source_claim=source_claim)
                for fact in facts
                if fact.strip()
            ]

        except json.JSONDecodeError as e:
            logger.warning(
                "failed_to_parse_decomposition",
                error=str(e),
                response=response_text[:200],
            )
            # Fallback: return original claim as single fact
            return [AtomicFact(text=source_claim, source_claim=source_claim)]

    async def decompose_batch(
        self,
        claims: list[str],
    ) -> dict[str, list[AtomicFact]]:
        """Decompose multiple claims.

        Returns dict mapping claim text -> atomic facts
        """
        results = {}
        for claim in claims:
            facts = await self.decompose(claim)
            results[claim] = facts
        return results


class RuleBasedDecomposer:
    """Simple rule-based decomposition for common patterns.

    Faster and cheaper than LLM-based decomposition for simple cases.
    Use as a first pass before falling back to LLM decomposition.
    """

    COMPOUND_PATTERNS: ClassVar[list[str]] = [
        r"\band\b",
        r"\bwhich\b",
        r"\bthat\b",
        r",\s*(?:and|or)\s*",
        r";\s*",
    ]

    def is_atomic(self, claim_text: str) -> bool:
        """Check if a claim is likely already atomic."""
        for pattern in self.COMPOUND_PATTERNS:
            if re.search(pattern, claim_text, re.IGNORECASE):
                return False
        return True

    def simple_decompose(self, claim_text: str) -> list[AtomicFact]:
        """Attempt simple rule-based decomposition.

        Returns None if claim is too complex for rule-based decomposition.
        """
        if self.is_atomic(claim_text):
            return [AtomicFact(text=claim_text, source_claim=claim_text)]

        # Try to split on common patterns
        facts = []

        # Split on "and"
        parts = re.split(r"\s+and\s+", claim_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if part:
                    facts.append(AtomicFact(text=part, source_claim=claim_text))
            return facts

        # If we can't decompose, return original
        return [AtomicFact(text=claim_text, source_claim=claim_text)]


class HybridDecomposer:
    """Hybrid decomposer that uses rules for simple cases, LLM for complex."""

    def __init__(
        self,
        llm_decomposer: ClaimDecomposer | None = None,
        complexity_threshold: int = 2,  # Number of compound patterns to trigger LLM
    ):
        self.rule_based = RuleBasedDecomposer()
        self.llm_decomposer = llm_decomposer
        self.complexity_threshold = complexity_threshold

    def _count_compound_patterns(self, claim_text: str) -> int:
        """Count compound patterns in claim."""
        count = 0
        for pattern in RuleBasedDecomposer.COMPOUND_PATTERNS:
            if re.search(pattern, claim_text, re.IGNORECASE):
                count += 1
        return count

    async def decompose(self, claim_text: str) -> list[AtomicFact]:
        """Decompose claim using appropriate method."""
        complexity = self._count_compound_patterns(claim_text)

        if complexity < self.complexity_threshold:
            # Use rule-based for simple cases
            return self.rule_based.simple_decompose(claim_text)

        # Use LLM for complex cases
        if self.llm_decomposer:
            return await self.llm_decomposer.decompose(claim_text)

        # Fallback to rule-based if no LLM
        return self.rule_based.simple_decompose(claim_text)

    def decompose_sync(self, claim_text: str) -> list[AtomicFact]:
        """Synchronous decomposition using rules only."""
        return self.rule_based.simple_decompose(claim_text)
