"""Comparison Table - Build feature comparison tables from retrieved content."""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from src.retrieval.simple_retriever import RetrievalResult
from src.storage.pg_vector_store import SearchResult

logger = structlog.get_logger()


@dataclass
class ComparisonRow:
    """A single row in a comparison table."""

    feature: str
    values: dict[str, str]
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "values": self.values,
            "sources": self.sources,
        }


@dataclass
class ComparisonTable:
    """A structured comparison of multiple entities."""

    entities: list[str]
    rows: list[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "rows": [r.to_dict() for r in self.rows],
            "feature_count": len(self.rows),
        }

    def to_markdown(self) -> str:
        """Render the comparison table as markdown."""
        if not self.entities or not self.rows:
            return "No comparison data available."

        header = "| Feature | " + " | ".join(self.entities) + " |"
        separator = "|" + "|".join(["---"] * (len(self.entities) + 1)) + "|"

        lines = [header, separator]
        for row in self.rows:
            values = [row.values.get(entity, "N/A") for entity in self.entities]
            line = f"| {row.feature} | " + " | ".join(values) + " |"
            lines.append(line)

        return "\n".join(lines)


class ComparisonBuilder:
    """Build feature comparison tables from retrieved content.

    Uses LLM to identify comparison dimensions and extract values for each entity.
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

    async def build(
        self,
        entities: list[str],
        retrieval_results: list[SearchResult] | RetrievalResult,
    ) -> ComparisonTable:
        """Build a comparison table for the given entities.

        Args:
            entities: List of items to compare (e.g., ["EIP-1559", "EIP-4844"])
            retrieval_results: Retrieved content to extract comparison data from

        Returns:
            ComparisonTable with features and values for each entity
        """
        if isinstance(retrieval_results, RetrievalResult):
            results = retrieval_results.results
        else:
            results = retrieval_results

        if not entities:
            return ComparisonTable(entities=[], rows=[])

        if not results:
            return ComparisonTable(entities=entities, rows=[])

        context = self._format_context(results)

        dimensions = await self.identify_comparison_dimensions(entities, results)

        rows = await self._extract_comparison_values(entities, dimensions, context, results)

        logger.info(
            "built_comparison_table",
            entities=entities,
            dimensions=len(dimensions),
            rows=len(rows),
        )

        return ComparisonTable(entities=entities, rows=rows)

    async def identify_comparison_dimensions(
        self,
        entities: list[str],
        content: list[SearchResult],
    ) -> list[str]:
        """Auto-detect comparison dimensions from content.

        Args:
            entities: Items being compared
            content: Retrieved content to analyze

        Returns:
            List of dimension/feature names relevant for comparison
        """
        context = self._format_context(content)
        entities_str = ", ".join(entities)

        prompt = f"""Analyze the following content about {entities_str} and identify the key dimensions/features that would be useful for comparison.

Focus on:
- Technical specifications (gas costs, block sizes, etc.)
- Design goals and motivations
- Implementation details
- Trade-offs and limitations
- Status and adoption

<context>
{context}
</context>

List 5-10 key comparison dimensions as a JSON array of strings.
Example: ["Gas cost", "Block size impact", "Implementation complexity", "Backwards compatibility"]

Respond ONLY with the JSON array.
"""

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            return self._default_dimensions()

        try:
            dimensions = json.loads(json_match.group())
            if isinstance(dimensions, list):
                return [str(d) for d in dimensions if d]
        except json.JSONDecodeError:
            pass

        return self._default_dimensions()

    def _default_dimensions(self) -> list[str]:
        """Return default comparison dimensions for EIPs."""
        return [
            "Purpose",
            "Status",
            "Type",
            "Gas impact",
            "Backwards compatibility",
            "Key changes",
        ]

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

    async def _extract_comparison_values(
        self,
        entities: list[str],
        dimensions: list[str],
        context: str,
        results: list[SearchResult],
    ) -> list[ComparisonRow]:
        """Extract comparison values for each entity and dimension."""
        entities_str = ", ".join(entities)
        dimensions_str = ", ".join(dimensions)

        prompt = f"""Compare {entities_str} across the following dimensions: {dimensions_str}

Based on the provided context, create a comparison table.

For each dimension, provide the value or description for each entity.
If information is not available, use "N/A".
Keep values concise (1-2 sentences max).

<context>
{context}
</context>

Format your response as a JSON array where each object has:
- "feature": the dimension name
- "values": an object mapping entity name to its value
- "sources": array of source numbers used

Example:
[
  {{"feature": "Gas cost", "values": {{"EIP-1559": "Base fee + tip", "EIP-4844": "Blob gas market"}}, "sources": ["1", "2"]}},
  ...
]

Respond ONLY with the JSON array.
"""

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        rows = self._parse_comparison_response(response_text, entities, results)
        return rows

    def _parse_comparison_response(
        self,
        response_text: str,
        entities: list[str],
        results: list[SearchResult],
    ) -> list[ComparisonRow]:
        """Parse LLM response into ComparisonRow objects."""
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            return []

        try:
            rows_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_comparison_response", response=response_text[:200])
            return []

        rows = []
        for item in rows_data:
            if not isinstance(item, dict):
                continue

            feature = item.get("feature", "")
            values = item.get("values", {})
            source_refs = item.get("sources", [])

            if not feature or not values:
                continue

            if not isinstance(values, dict):
                continue

            sources = []
            for ref in source_refs:
                ref_match = re.search(r"(\d+)", str(ref))
                if ref_match:
                    idx = int(ref_match.group(1)) - 1
                    if 0 <= idx < len(results):
                        sources.append(results[idx].chunk.document_id)

            rows.append(
                ComparisonRow(
                    feature=feature,
                    values=values,
                    sources=sources,
                )
            )

        return rows
