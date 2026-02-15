"""Structured Generator - Orchestrate generation of structured outputs."""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import anthropic
import structlog

from src.config import DEFAULT_MODEL
from src.graph.dependency_traverser import DependencyTraverser
from src.graph.falkordb_store import FalkorDBStore
from src.retrieval.simple_retriever import RetrievalResult, SimpleRetriever
from src.structured.argument_mapper import ArgumentMap, ArgumentMapper
from src.structured.comparison_table import ComparisonBuilder, ComparisonTable
from src.structured.dependency_view import DependencyView, DependencyViewBuilder
from src.structured.timeline_builder import Timeline, TimelineBuilder

logger = structlog.get_logger()


class StructuredOutputType(Enum):
    """Types of structured outputs."""

    TIMELINE = "timeline"
    ARGUMENT_MAP = "argument_map"
    COMPARISON = "comparison"
    DEPENDENCY = "dependency"


@dataclass
class StructuredOutput:
    """A structured output with type and data."""

    output_type: StructuredOutputType
    query: str
    data: Timeline | ArgumentMap | ComparisonTable | DependencyView
    sources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "query": self.query,
            "data": self.data.to_dict(),
            "sources": self.sources,
        }


class StructuredGenerator:
    """Generate structured outputs from queries.

    Orchestrates the appropriate builder based on query type or explicit request.
    """

    TYPE_INDICATORS: ClassVar[dict[StructuredOutputType, list[str]]] = {
        StructuredOutputType.TIMELINE: [
            "timeline",
            "history",
            "evolution",
            "when",
            "chronolog",
            "over time",
            "progression",
            "development of",
        ],
        StructuredOutputType.ARGUMENT_MAP: [
            "pro",
            "con",
            "debate",
            "argument",
            "opinion",
            "controversy",
            "tradeoff",
            "trade-off",
            "pros and cons",
            "advantages",
            "disadvantages",
        ],
        StructuredOutputType.COMPARISON: [
            "compare",
            "comparison",
            "difference",
            "vs",
            "versus",
            "contrast",
            "between",
            "similarities",
        ],
        StructuredOutputType.DEPENDENCY: [
            "depend",
            "require",
            "relationship",
            "relies on",
            "related to",
            "builds on",
            "based on",
            "prerequisite",
        ],
    }

    def __init__(
        self,
        retriever: SimpleRetriever,
        graph_store: FalkorDBStore | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.retriever = retriever
        self.graph_store = graph_store

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        self.timeline_builder = TimelineBuilder(api_key=self.api_key, model=model)
        self.argument_mapper = ArgumentMapper(api_key=self.api_key, model=model)
        self.comparison_builder = ComparisonBuilder(api_key=self.api_key, model=model)

        self.dependency_builder: DependencyViewBuilder | None = None
        if graph_store:
            traverser = DependencyTraverser(graph_store)
            self.dependency_builder = DependencyViewBuilder(graph_store, traverser)

    async def generate(
        self,
        query: str,
        output_type: StructuredOutputType | None = None,
        top_k: int = 15,
    ) -> StructuredOutput:
        """Generate a structured output for the query.

        Args:
            query: The user's question or request
            output_type: Explicitly specify output type, or auto-detect if None
            top_k: Number of chunks to retrieve

        Returns:
            StructuredOutput with the appropriate data structure
        """
        if output_type is None:
            output_type = self.auto_detect_type(query)

        logger.info(
            "generating_structured_output",
            query=query[:50],
            output_type=output_type.value,
        )

        if output_type == StructuredOutputType.DEPENDENCY:
            return await self._generate_dependency(query)

        retrieval = await self.retriever.retrieve(query=query, top_k=top_k)
        sources = self._collect_sources(retrieval)

        if output_type == StructuredOutputType.TIMELINE:
            data = await self.timeline_builder.build(query, retrieval)

        elif output_type == StructuredOutputType.ARGUMENT_MAP:
            data = await self.argument_mapper.map_arguments(query, retrieval)

        elif output_type == StructuredOutputType.COMPARISON:
            entities = self._extract_comparison_entities(query)
            data = await self.comparison_builder.build(entities, retrieval)

        else:
            data = await self.timeline_builder.build(query, retrieval)

        return StructuredOutput(
            output_type=output_type,
            query=query,
            data=data,
            sources=sources,
        )

    def auto_detect_type(self, query: str) -> StructuredOutputType:
        """Auto-detect the appropriate output type from query.

        Args:
            query: The user's question

        Returns:
            The most appropriate StructuredOutputType
        """
        query_lower = query.lower()

        scores: dict[StructuredOutputType, int] = {t: 0 for t in StructuredOutputType}

        for output_type, indicators in self.TYPE_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    scores[output_type] += 1
                    if indicator in ["compare", "vs", "versus", "comparison"]:
                        scores[output_type] += 2
                    if indicator in ["timeline", "history", "chronolog"]:
                        scores[output_type] += 2
                    if indicator in ["pros and cons", "debate", "argument"]:
                        scores[output_type] += 2

        eip_pattern = r"eip[- ]?\d+"
        eip_matches = re.findall(eip_pattern, query_lower)
        if len(eip_matches) >= 2:
            scores[StructuredOutputType.COMPARISON] += 3

        if len(eip_matches) == 1 and any(
            word in query_lower for word in ["depend", "require", "related"]
        ):
            scores[StructuredOutputType.DEPENDENCY] += 3

        best_type = max(scores, key=lambda t: scores[t])

        if scores[best_type] == 0:
            return StructuredOutputType.TIMELINE

        return best_type

    async def _generate_dependency(self, query: str) -> StructuredOutput:
        """Generate a dependency view from query."""
        if not self.dependency_builder:
            logger.warning("dependency_builder_not_available")
            retrieval = await self.retriever.retrieve(query=query, top_k=10)
            data = await self.timeline_builder.build(query, retrieval)
            return StructuredOutput(
                output_type=StructuredOutputType.TIMELINE,
                query=query,
                data=data,
                sources=self._collect_sources(retrieval),
            )

        eip_number = self._extract_eip_number(query)

        if eip_number is None:
            retrieval = await self.retriever.retrieve(query=query, top_k=10)
            data = await self.timeline_builder.build(query, retrieval)
            return StructuredOutput(
                output_type=StructuredOutputType.TIMELINE,
                query=query,
                data=data,
                sources=self._collect_sources(retrieval),
            )

        depth = 3
        if "deep" in query.lower() or "all" in query.lower():
            depth = 5
        elif "direct" in query.lower():
            depth = 1

        data = self.dependency_builder.build(eip_number=eip_number, depth=depth)

        sources = [
            {"document_id": f"eip-{eip_number}", "type": "graph"}
            for node in data.nodes
        ]

        return StructuredOutput(
            output_type=StructuredOutputType.DEPENDENCY,
            query=query,
            data=data,
            sources=sources[:10],
        )

    def _extract_eip_number(self, query: str) -> int | None:
        """Extract an EIP number from query."""
        eip_pattern = r"eip[- ]?(\d+)"
        match = re.search(eip_pattern, query.lower())
        if match:
            return int(match.group(1))
        return None

    def _extract_comparison_entities(self, query: str) -> list[str]:
        """Extract entities to compare from query."""
        entities = []

        eip_pattern = r"eip[- ]?(\d+)"
        eip_matches = re.findall(eip_pattern, query.lower())
        for match in eip_matches:
            entities.append(f"EIP-{match}")

        if len(entities) >= 2:
            return entities

        common_entities = [
            ("proof of work", "PoW"),
            ("proof of stake", "PoS"),
            ("eip-1559", "EIP-1559"),
            ("eip-4844", "EIP-4844"),
            ("layer 1", "L1"),
            ("layer 2", "L2"),
            ("optimistic rollup", "Optimistic Rollups"),
            ("zk rollup", "ZK Rollups"),
        ]

        query_lower = query.lower()
        for term, label in common_entities:
            if term in query_lower and label not in entities:
                entities.append(label)

        if len(entities) < 2:
            vs_pattern = r"(\w+(?:\s+\w+)?)\s+(?:vs\.?|versus|or)\s+(\w+(?:\s+\w+)?)"
            match = re.search(vs_pattern, query, re.IGNORECASE)
            if match:
                entities = [match.group(1).strip(), match.group(2).strip()]

        return entities if len(entities) >= 2 else ["Entity A", "Entity B"]

    def _collect_sources(self, retrieval: RetrievalResult) -> list[dict[str, Any]]:
        """Collect unique sources from retrieval results."""
        seen = set()
        sources = []

        for result in retrieval.results:
            doc_id = result.chunk.document_id
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append(
                    {
                        "document_id": doc_id,
                        "section": result.chunk.section_path,
                        "similarity": result.similarity,
                    }
                )

        return sources
