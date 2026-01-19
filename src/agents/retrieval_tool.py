"""Retrieval Tool - Agent interface to retrieval subsystem."""

from dataclasses import dataclass, field
from enum import Enum

import structlog

from src.filters.metadata_filter import MetadataQuery
from src.graph.dependency_traverser import DependencyTraverser, TraversalDirection
from src.graph.falkordb_store import FalkorDBStore
from src.retrieval.hybrid_retriever import HybridResult, HybridRetriever
from src.retrieval.simple_retriever import SimpleRetriever
from src.storage.pg_vector_store import SearchResult

logger = structlog.get_logger()


class RetrievalMode(Enum):
    """Available retrieval modes for the agent."""

    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    query: str
    mode: RetrievalMode
    chunks: list[dict]
    total_tokens: int
    metadata: dict = field(default_factory=dict)

    def format_context(self, max_chunks: int | None = None) -> str:
        """Format chunks as context string for LLM consumption."""
        chunks_to_format = self.chunks[:max_chunks] if max_chunks else self.chunks
        parts = []

        for i, chunk in enumerate(chunks_to_format):
            source = chunk.get("document_id", "unknown").upper()
            section = chunk.get("section_path")
            if section:
                source = f"{source} ({section})"

            content = chunk.get("content", "")
            parts.append(f"[{i + 1}] [{source}]\n{content}")

        return "\n\n---\n\n".join(parts)


@dataclass
class RetrievalFilters:
    """Filters that can be applied to retrieval."""

    document_ids: list[str] | None = None
    statuses: list[str] | None = None
    types: list[str] | None = None
    categories: list[str] | None = None
    eip_numbers: list[int] | None = None

    def to_metadata_query(self) -> MetadataQuery:
        """Convert to MetadataQuery for hybrid retriever."""
        return MetadataQuery(
            statuses=self.statuses or [],
            types=self.types or [],
            categories=self.categories or [],
            eip_numbers=self.eip_numbers or [],
        )


class RetrievalTool:
    """Tool for agent to perform retrieval operations.

    Wraps existing retrievers and provides a unified interface for:
    - Vector search (semantic similarity)
    - Hybrid search (BM25 + vector with RRF)
    - Graph traversal (EIP relationships)
    """

    def __init__(
        self,
        simple_retriever: SimpleRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        graph_store: FalkorDBStore | None = None,
        default_limit: int = 5,
    ):
        self.simple_retriever = simple_retriever
        self.hybrid_retriever = hybrid_retriever
        self.graph_store = graph_store
        self.default_limit = default_limit

        if graph_store:
            self.traverser = DependencyTraverser(graph_store)
        else:
            self.traverser = None

    async def execute(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        limit: int | None = None,
    ) -> RetrievalResult:
        """Execute a retrieval operation.

        Args:
            query: The search query
            filters: Optional filters to apply
            mode: Retrieval mode to use
            limit: Maximum number of results

        Returns:
            RetrievalResult with retrieved chunks
        """
        k = limit or self.default_limit

        if mode == RetrievalMode.VECTOR:
            return await self._vector_search(query, filters, k)
        elif mode == RetrievalMode.HYBRID:
            return await self._hybrid_search(query, filters, k)
        elif mode == RetrievalMode.GRAPH:
            return await self._graph_search(query, filters, k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    async def _vector_search(
        self,
        query: str,
        filters: RetrievalFilters | None,
        limit: int,
    ) -> RetrievalResult:
        """Perform vector similarity search."""
        if not self.simple_retriever:
            raise ValueError("SimpleRetriever not configured")

        document_filter = None
        if filters and filters.document_ids:
            document_filter = filters.document_ids[0]

        result = await self.simple_retriever.retrieve(
            query=query,
            top_k=limit,
            document_filter=document_filter,
        )

        chunks = self._convert_search_results(result.results)

        logger.debug(
            "vector_search_complete",
            query=query[:50],
            num_results=len(chunks),
            total_tokens=result.total_tokens,
        )

        return RetrievalResult(
            query=query,
            mode=RetrievalMode.VECTOR,
            chunks=chunks,
            total_tokens=result.total_tokens,
            metadata={"similarity_range": self._get_similarity_range(result.results)},
        )

    async def _hybrid_search(
        self,
        query: str,
        filters: RetrievalFilters | None,
        limit: int,
    ) -> RetrievalResult:
        """Perform hybrid BM25 + vector search."""
        if not self.hybrid_retriever:
            if self.simple_retriever:
                logger.warning("hybrid_retriever_not_available_falling_back_to_vector")
                return await self._vector_search(query, filters, limit)
            raise ValueError("No retriever configured")

        metadata_query = filters.to_metadata_query() if filters else None

        result = await self.hybrid_retriever.retrieve(
            query=query,
            limit=limit,
            metadata_query=metadata_query,
        )

        chunks = self._convert_hybrid_results(result.results)

        logger.debug(
            "hybrid_search_complete",
            query=query[:50],
            num_results=len(chunks),
            total_tokens=result.total_tokens,
            vector_count=result.vector_results_count,
            bm25_count=result.bm25_results_count,
        )

        return RetrievalResult(
            query=query,
            mode=RetrievalMode.HYBRID,
            chunks=chunks,
            total_tokens=result.total_tokens,
            metadata={
                "vector_results": result.vector_results_count,
                "bm25_results": result.bm25_results_count,
            },
        )

    async def _graph_search(
        self,
        query: str,
        filters: RetrievalFilters | None,
        limit: int,
    ) -> RetrievalResult:
        """Perform graph-augmented search using EIP relationships."""
        if not self.traverser or not self.simple_retriever:
            logger.warning("graph_search_not_available_falling_back_to_vector")
            return await self._vector_search(query, filters, limit)

        eip_numbers = self._extract_eip_numbers(query)
        if filters and filters.eip_numbers:
            eip_numbers.update(filters.eip_numbers)

        related_eips: set[int] = set()
        for eip_num in eip_numbers:
            deps = self.traverser.get_dependencies(
                eip_num,
                direction=TraversalDirection.BOTH,
                max_depth=2,
            )
            related_eips.update(deps.get_related_eips())

        primary_results = await self.simple_retriever.retrieve(query=query, top_k=limit)
        chunks = self._convert_search_results(primary_results.results)
        total_tokens = primary_results.total_tokens

        related_chunks = []
        for eip_num in list(related_eips)[:3]:
            doc_id = f"eip-{eip_num}"
            related_result = await self.simple_retriever.retrieve(
                query=query,
                top_k=2,
                document_filter=doc_id,
            )
            related_chunks.extend(self._convert_search_results(related_result.results))
            total_tokens += related_result.total_tokens

        seen_ids = {c["chunk_id"] for c in chunks}
        for chunk in related_chunks:
            if chunk["chunk_id"] not in seen_ids:
                chunks.append(chunk)
                seen_ids.add(chunk["chunk_id"])

        logger.debug(
            "graph_search_complete",
            query=query[:50],
            mentioned_eips=list(eip_numbers),
            related_eips=list(related_eips)[:5],
            num_results=len(chunks),
            total_tokens=total_tokens,
        )

        return RetrievalResult(
            query=query,
            mode=RetrievalMode.GRAPH,
            chunks=chunks[: limit + 5],
            total_tokens=total_tokens,
            metadata={
                "mentioned_eips": list(eip_numbers),
                "related_eips": list(related_eips),
            },
        )

    def _convert_search_results(self, results: list[SearchResult]) -> list[dict]:
        """Convert SearchResult objects to dict format."""
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.document_id,
                "content": r.chunk.content,
                "token_count": r.chunk.token_count,
                "section_path": r.chunk.section_path,
                "similarity": r.similarity,
            }
            for r in results
        ]

    def _convert_hybrid_results(self, results: list[HybridResult]) -> list[dict]:
        """Convert HybridResult objects to dict format."""
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.document_id,
                "content": r.chunk.content,
                "token_count": r.chunk.token_count,
                "section_path": r.chunk.section_path,
                "rrf_score": r.rrf_score,
                "vector_rank": r.vector_rank,
                "bm25_rank": r.bm25_rank,
            }
            for r in results
        ]

    def _get_similarity_range(self, results: list[SearchResult]) -> dict[str, float]:
        """Get min/max similarity scores from results."""
        if not results:
            return {"min": 0.0, "max": 0.0}

        similarities = [r.similarity for r in results]
        return {"min": min(similarities), "max": max(similarities)}

    def _extract_eip_numbers(self, text: str) -> set[int]:
        """Extract EIP numbers mentioned in text."""
        import re

        matches = re.findall(r"eip[- ]?(\d+)", text, re.IGNORECASE)
        return set(int(m) for m in matches)

    def get_available_modes(self) -> list[RetrievalMode]:
        """Get list of available retrieval modes based on configuration."""
        modes = []
        if self.simple_retriever:
            modes.append(RetrievalMode.VECTOR)
        if self.hybrid_retriever:
            modes.append(RetrievalMode.HYBRID)
        if self.traverser and self.simple_retriever:
            modes.append(RetrievalMode.GRAPH)
        return modes
