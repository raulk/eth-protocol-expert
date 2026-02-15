"""Timeline Builder - Build chronological views from retrieved content."""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import anthropic
import structlog

from src.config import DEFAULT_MODEL
from src.retrieval.simple_retriever import RetrievalResult
from src.storage.pg_vector_store import SearchResult

logger = structlog.get_logger()


@dataclass
class TimelineEvent:
    """A single event in a timeline."""

    date: str
    event: str
    source_id: str
    source_type: str
    confidence: float


@dataclass
class Timeline:
    """A chronological sequence of events."""

    title: str
    events: list[TimelineEvent] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "events": [
                {
                    "date": e.date,
                    "event": e.event,
                    "source_id": e.source_id,
                    "source_type": e.source_type,
                    "confidence": e.confidence,
                }
                for e in self.events
            ],
            "sources": self.sources,
        }


class TimelineBuilder:
    """Build chronological views from retrieved content.

    Uses LLM to extract dated events from source material and organizes
    them into a coherent timeline.
    """

    DATE_PATTERNS: ClassVar[list[str]] = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4}/\d{2}/\d{2})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        r"\b(Q[1-4]\s+\d{4})\b",
        r"\b(\d{4})\b",
    ]

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

    async def build(
        self,
        query: str,
        retrieval_results: list[SearchResult] | RetrievalResult,
    ) -> Timeline:
        """Build a timeline from retrieved content.

        Args:
            query: The original query that prompted the timeline
            retrieval_results: Retrieved content to extract events from

        Returns:
            Timeline with chronologically ordered events
        """
        if isinstance(retrieval_results, RetrievalResult):
            results = retrieval_results.results
        else:
            results = retrieval_results

        if not results:
            return Timeline(title=f"Timeline: {query}", events=[], sources=[])

        context = self._format_context(results)
        extracted_events = await self._extract_events_with_llm(query, context, results)

        sorted_events = self.sort_chronologically(extracted_events)

        sources = self._collect_sources(results)

        logger.info(
            "built_timeline",
            query=query[:50],
            num_events=len(sorted_events),
            num_sources=len(sources),
        )

        return Timeline(
            title=f"Timeline: {query}",
            events=sorted_events,
            sources=sources,
        )

    def extract_dates(self, text: str) -> list[tuple[str, str]]:
        """Extract dates and their surrounding context from text.

        Returns:
            List of (date_string, context) tuples
        """
        results = []
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_str = match.group(1)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                results.append((date_str, context))

        return results

    def sort_chronologically(
        self,
        events: list[TimelineEvent],
    ) -> list[TimelineEvent]:
        """Sort events by date, handling various date formats."""

        def parse_date_for_sort(date_str: str) -> tuple[int, int, int]:
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    return (dt.year, dt.month, dt.day)
                except ValueError:
                    pass

            if re.match(r"\d{4}/\d{2}/\d{2}", date_str):
                try:
                    dt = datetime.strptime(date_str, "%Y/%m/%d")
                    return (dt.year, dt.month, dt.day)
                except ValueError:
                    pass

            for fmt in ["%B %d, %Y", "%B %d %Y", "%d %B %Y"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return (dt.year, dt.month, dt.day)
                except ValueError:
                    continue

            quarter_match = re.match(r"Q(\d)\s+(\d{4})", date_str)
            if quarter_match:
                quarter = int(quarter_match.group(1))
                year = int(quarter_match.group(2))
                month = (quarter - 1) * 3 + 1
                return (year, month, 1)

            year_match = re.match(r"(\d{4})", date_str)
            if year_match:
                return (int(year_match.group(1)), 1, 1)

            return (9999, 12, 31)

        return sorted(events, key=lambda e: parse_date_for_sort(e.date))

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

    async def _extract_events_with_llm(
        self,
        query: str,
        context: str,
        results: list[SearchResult],
    ) -> list[TimelineEvent]:
        """Use LLM to extract dated events from context."""
        prompt = f"""Analyze the following content and extract events with dates that are relevant to the query: "{query}"

For each event, provide:
1. The date (in the format found in the source, or YYYY-MM-DD if you can normalize it)
2. A brief description of the event
3. The source number it came from (e.g., "Source 1")
4. Your confidence level (high/medium/low)

Format your response as a JSON array of objects with keys: date, event, source, confidence

Only include events that have specific dates or time references mentioned in the sources.
If the source mentions a quarter (Q1, Q2, etc.) or just a year, use that as the date.

<context>
{context}
</context>

Respond ONLY with a valid JSON array. If no dated events are found, respond with [].
"""

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        events = self._parse_llm_response(response_text, results)
        return events

    def _parse_llm_response(
        self,
        response_text: str,
        results: list[SearchResult],
    ) -> list[TimelineEvent]:
        """Parse LLM response into TimelineEvent objects."""
        import json

        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            return []

        try:
            events_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_timeline_response", response=response_text[:200])
            return []

        events = []
        for item in events_data:
            if not isinstance(item, dict):
                continue

            date = item.get("date", "")
            event_text = item.get("event", "")
            source_ref = item.get("source", "")
            confidence_str = item.get("confidence", "medium")

            if not date or not event_text:
                continue

            source_num_match = re.search(r"(\d+)", str(source_ref))
            source_idx = int(source_num_match.group(1)) - 1 if source_num_match else 0
            source_idx = max(0, min(source_idx, len(results) - 1))

            source_result = results[source_idx] if results else None
            source_id = source_result.chunk.document_id if source_result else "unknown"
            source_type = self._determine_source_type(source_id)

            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            confidence = confidence_map.get(str(confidence_str).lower(), 0.7)

            events.append(
                TimelineEvent(
                    date=date,
                    event=event_text,
                    source_id=source_id,
                    source_type=source_type,
                    confidence=confidence,
                )
            )

        return events

    def _determine_source_type(self, source_id: str) -> str:
        """Determine the type of source based on document ID."""
        source_lower = source_id.lower()
        if source_lower.startswith("eip-"):
            return "eip"
        if "magicians" in source_lower or "forum" in source_lower:
            return "forum"
        if "research" in source_lower:
            return "research"
        return "document"

    def _collect_sources(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Collect unique sources from results."""
        seen = set()
        sources = []

        for result in results:
            doc_id = result.chunk.document_id
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append(
                    {
                        "document_id": doc_id,
                        "section": result.chunk.section_path,
                        "type": self._determine_source_type(doc_id),
                    }
                )

        return sources
