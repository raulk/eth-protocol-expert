"""Reflection - Agent self-assessment of retrieval sufficiency."""

from dataclasses import dataclass

import structlog

from ..config import DEFAULT_MODEL
from ..generation.completion import call_llm

logger = structlog.get_logger()


@dataclass
class ReflectionResult:
    """Result of reflection on retrieval sufficiency."""

    sufficient: bool
    missing_info: list[str]
    confidence: float
    reasoning: str


class Reflector:
    """Assess whether retrieved information is sufficient to answer a query.

    The reflector asks: "Did I find what I needed?"

    Uses an LLM to analyze:
    - Coverage: Does retrieved info address the query?
    - Completeness: Are there obvious gaps?
    - Confidence: How certain is the assessment?
    """

    REFLECTION_PROMPT = """You are evaluating whether retrieved information is sufficient to answer a question about Ethereum protocol.

<query>
{query}
</query>

<agent_thoughts>
{thoughts}
</agent_thoughts>

<retrieved_information>
{retrieved}
</retrieved_information>

Analyze whether the retrieved information is sufficient to provide a complete, accurate answer.

Consider:
1. Does the information directly address the query?
2. Are there obvious gaps or missing pieces?
3. Is there contradictory information that needs resolution?
4. Would additional retrieval likely help?

Respond with a JSON object:
{{
    "sufficient": true/false,
    "missing_info": ["list of specific missing pieces, if any"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of assessment"
}}

Only respond with the JSON object, no other text."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.7,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold

    async def reflect(
        self,
        query: str,
        retrieved: list[dict],
        thoughts: list[str],
    ) -> ReflectionResult:
        """Reflect on whether retrieved information is sufficient.

        Args:
            query: The original user query
            retrieved: List of retrieved chunks
            thoughts: Agent's reasoning so far

        Returns:
            ReflectionResult with assessment
        """
        retrieved_text = self._format_retrieved(retrieved)
        thoughts_text = "\n".join(f"- {t}" for t in thoughts) if thoughts else "No thoughts yet."

        prompt = self.REFLECTION_PROMPT.format(
            query=query,
            thoughts=thoughts_text,
            retrieved=retrieved_text,
        )

        completion = await call_llm(
            self.model, [{"role": "user", "content": prompt}], max_tokens=512
        )
        result = self._parse_response(completion.text)

        logger.debug(
            "reflection_complete",
            query=query[:50],
            sufficient=result.sufficient,
            confidence=result.confidence,
            missing_count=len(result.missing_info),
        )

        return result

    def _format_retrieved(self, retrieved: list[dict]) -> str:
        """Format retrieved chunks for the prompt."""
        if not retrieved:
            return "No information retrieved yet."

        parts = []
        for i, chunk in enumerate(retrieved):
            source = chunk.get("document_id", "unknown").upper()
            section = chunk.get("section_path")
            if section:
                source = f"{source} ({section})"

            content = chunk.get("content", "")[:500]
            parts.append(f"[{i + 1}] [{source}]\n{content}")

        return "\n\n".join(parts)

    def _parse_response(self, response_text: str) -> ReflectionResult:
        """Parse LLM response into ReflectionResult."""
        import json

        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            return ReflectionResult(
                sufficient=data.get("sufficient", False),
                missing_info=data.get("missing_info", []),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("reflection_parse_error", error=str(e), response=response_text[:200])
            return ReflectionResult(
                sufficient=False,
                missing_info=["Unable to parse reflection response"],
                confidence=0.0,
                reasoning=f"Parse error: {e}",
            )

    def is_confident(self, result: ReflectionResult) -> bool:
        """Check if reflection result meets confidence threshold."""
        return result.sufficient and result.confidence >= self.confidence_threshold

    def get_suggested_queries(self, result: ReflectionResult) -> list[str]:
        """Generate suggested follow-up queries from missing info.

        Args:
            result: Reflection result with missing_info

        Returns:
            List of suggested follow-up queries
        """
        if not result.missing_info:
            return []

        suggested = []
        for missing in result.missing_info[:3]:
            if "specific" in missing.lower() or "detail" in missing.lower():
                suggested.append(f"What are the specific details of {missing}?")
            elif "example" in missing.lower():
                suggested.append(f"Show examples of {missing}")
            elif "how" in missing.lower():
                suggested.append(missing)
            else:
                suggested.append(f"Explain {missing}")

        return suggested
