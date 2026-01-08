"""Argument Mapper - Map arguments into pro/con/neutral structure."""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic
import structlog

from src.retrieval.simple_retriever import RetrievalResult
from src.storage.pg_vector_store import SearchResult

logger = structlog.get_logger()


class Position(Enum):
    """Position of an argument."""

    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"


class Strength(Enum):
    """Strength of an argument."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class Argument:
    """A single argument with position and evidence."""

    position: Position
    claim: str
    evidence: str
    source_id: str
    strength: Strength

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.value,
            "claim": self.claim,
            "evidence": self.evidence,
            "source_id": self.source_id,
            "strength": self.strength.value,
        }


@dataclass
class ArgumentMap:
    """A structured map of arguments on a topic."""

    topic: str
    pro_arguments: list[Argument] = field(default_factory=list)
    con_arguments: list[Argument] = field(default_factory=list)
    neutral_points: list[Argument] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "pro_arguments": [a.to_dict() for a in self.pro_arguments],
            "con_arguments": [a.to_dict() for a in self.con_arguments],
            "neutral_points": [a.to_dict() for a in self.neutral_points],
            "summary": {
                "pro_count": len(self.pro_arguments),
                "con_count": len(self.con_arguments),
                "neutral_count": len(self.neutral_points),
            },
        }

    @property
    def all_arguments(self) -> list[Argument]:
        return self.pro_arguments + self.con_arguments + self.neutral_points


class ArgumentMapper:
    """Map arguments from retrieved content into structured pro/con format.

    Uses LLM to identify arguments, their positions, and supporting evidence.
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

    async def map_arguments(
        self,
        query: str,
        retrieval_results: list[SearchResult] | RetrievalResult,
    ) -> ArgumentMap:
        """Map arguments from retrieved content.

        Args:
            query: The topic or question to analyze
            retrieval_results: Retrieved content to extract arguments from

        Returns:
            ArgumentMap with pro, con, and neutral arguments
        """
        if isinstance(retrieval_results, RetrievalResult):
            results = retrieval_results.results
        else:
            results = retrieval_results

        if not results:
            return ArgumentMap(topic=query)

        context = self._format_context(results)
        arguments = await self._extract_arguments_with_llm(query, context, results)

        argument_map = ArgumentMap(topic=query)
        for arg in arguments:
            if arg.position == Position.PRO:
                argument_map.pro_arguments.append(arg)
            elif arg.position == Position.CON:
                argument_map.con_arguments.append(arg)
            else:
                argument_map.neutral_points.append(arg)

        argument_map.pro_arguments.sort(
            key=lambda a: self._strength_order(a.strength), reverse=True
        )
        argument_map.con_arguments.sort(
            key=lambda a: self._strength_order(a.strength), reverse=True
        )

        logger.info(
            "mapped_arguments",
            topic=query[:50],
            pro_count=len(argument_map.pro_arguments),
            con_count=len(argument_map.con_arguments),
            neutral_count=len(argument_map.neutral_points),
        )

        return argument_map

    def extract_positions(self, text: str) -> list[Argument]:
        """Extract positions from text using pattern matching.

        This is a simpler extraction method that can be used without LLM calls.
        """
        arguments = []

        pro_patterns = [
            r"(?:benefit|advantage|pro|support|favor)[\s:]+([^.]+\.)",
            r"(?:improves?|enables?|allows?)[\s]+([^.]+\.)",
        ]

        con_patterns = [
            r"(?:drawback|disadvantage|con|against|concern)[\s:]+([^.]+\.)",
            r"(?:risks?|issues?|problems?)[\s:]+([^.]+\.)",
        ]

        for pattern in pro_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = match.group(1).strip()
                if len(claim) > 20:
                    arguments.append(
                        Argument(
                            position=Position.PRO,
                            claim=claim,
                            evidence="",
                            source_id="pattern_match",
                            strength=Strength.MODERATE,
                        )
                    )

        for pattern in con_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = match.group(1).strip()
                if len(claim) > 20:
                    arguments.append(
                        Argument(
                            position=Position.CON,
                            claim=claim,
                            evidence="",
                            source_id="pattern_match",
                            strength=Strength.MODERATE,
                        )
                    )

        return arguments

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into context for LLM."""
        parts = []
        for i, result in enumerate(results):
            chunk = result.chunk
            source_info = chunk.document_id.upper()
            if chunk.section_path:
                source_info = f"{source_info} - {chunk.section_path}"

            parts.append(f"[Source {i + 1}: {source_info}]\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    async def _extract_arguments_with_llm(
        self,
        query: str,
        context: str,
        results: list[SearchResult],
    ) -> list[Argument]:
        """Use LLM to extract arguments from context."""
        prompt = f"""Analyze the following content and identify distinct arguments related to: "{query}"

For each argument, determine:
1. Position: Is it "pro" (supporting/favorable), "con" (opposing/critical), or "neutral" (factual observation)
2. Claim: The main assertion being made
3. Evidence: Key supporting evidence from the source
4. Source: Which source number it comes from
5. Strength: How strong is the argument (strong/moderate/weak)

Format your response as a JSON array of objects with keys: position, claim, evidence, source, strength

Guidelines:
- Extract arguments that people make FOR or AGAINST the topic
- Include technical trade-offs and design decisions
- Capture concerns raised in discussions
- Include neutral factual points that inform the debate

<context>
{context}
</context>

Respond ONLY with a valid JSON array. If no arguments are found, respond with [].
"""

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        arguments = self._parse_llm_response(response_text, results)
        return arguments

    def _parse_llm_response(
        self,
        response_text: str,
        results: list[SearchResult],
    ) -> list[Argument]:
        """Parse LLM response into Argument objects."""
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            return []

        try:
            args_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_argument_response", response=response_text[:200])
            return []

        arguments = []
        for item in args_data:
            if not isinstance(item, dict):
                continue

            position_str = item.get("position", "neutral").lower()
            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            source_ref = item.get("source", "")
            strength_str = item.get("strength", "moderate").lower()

            if not claim:
                continue

            position_map = {
                "pro": Position.PRO,
                "con": Position.CON,
                "neutral": Position.NEUTRAL,
            }
            position = position_map.get(position_str, Position.NEUTRAL)

            strength_map = {
                "strong": Strength.STRONG,
                "moderate": Strength.MODERATE,
                "weak": Strength.WEAK,
            }
            strength = strength_map.get(strength_str, Strength.MODERATE)

            source_num_match = re.search(r"(\d+)", str(source_ref))
            source_idx = int(source_num_match.group(1)) - 1 if source_num_match else 0
            source_idx = max(0, min(source_idx, len(results) - 1))

            source_result = results[source_idx] if results else None
            source_id = source_result.chunk.document_id if source_result else "unknown"

            arguments.append(
                Argument(
                    position=position,
                    claim=claim,
                    evidence=evidence,
                    source_id=source_id,
                    strength=strength,
                )
            )

        return arguments

    def _strength_order(self, strength: Strength) -> int:
        """Return numeric order for strength sorting."""
        order = {Strength.STRONG: 3, Strength.MODERATE: 2, Strength.WEAK: 1}
        return order.get(strength, 2)
