"""Graph-Augmented Retrieval - Enhance retrieval with EIP relationships."""

import re
from dataclasses import dataclass, field

import structlog

from ..embeddings.voyage_embedder import VoyageEmbedder
from ..graph.dependency_traverser import DependencyTraverser, TraversalDirection
from ..graph.falkordb_store import FalkorDBStore
from ..storage.pg_vector_store import PgVectorStore, SearchResult

logger = structlog.get_logger()


@dataclass
class GraphAugmentedResult:
    """Result from graph-augmented retrieval."""

    query: str

    # Primary results from vector search
    primary_results: list[SearchResult]

    # Related EIPs found via graph
    related_eips: set[int]

    # Results from related EIPs
    related_results: list[SearchResult]

    # Combined results (primary + related)
    combined_results: list[SearchResult]

    # Graph context
    dependency_context: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(r.chunk.token_count for r in self.combined_results)

    def get_unique_documents(self) -> list[str]:
        """Get list of unique document IDs in results."""
        docs = set()
        for r in self.combined_results:
            docs.add(r.chunk.document_id)
        return sorted(docs)


class GraphAugmentedRetriever:
    """Retriever that uses graph relationships to enhance results.

    When asked about EIP-X, this retriever:
    1. Performs standard vector search
    2. Identifies EIP mentions in the query
    3. Uses the graph to find related EIPs (dependencies, supersedes)
    4. Augments results with chunks from related EIPs
    """

    def __init__(
        self,
        embedder: VoyageEmbedder,
        vector_store: PgVectorStore,
        graph_store: FalkorDBStore,
        default_top_k: int = 10,
        related_top_k: int = 5,
        max_graph_depth: int = 2,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.traverser = DependencyTraverser(graph_store, max_depth=max_graph_depth)

        self.default_top_k = default_top_k
        self.related_top_k = related_top_k
        self.max_graph_depth = max_graph_depth

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        include_related: bool = True,
        max_related_eips: int = 5,
        related_depth: int | None = None,
    ) -> GraphAugmentedResult:
        """Retrieve relevant chunks with graph augmentation.

        Args:
            query: The user's question
            top_k: Number of primary results to retrieve
            include_related: Whether to include related EIP chunks
            max_related_eips: Maximum number of related EIPs to include
            related_depth: Graph traversal depth (default: max_graph_depth)

        Returns:
            GraphAugmentedResult with primary and related chunks
        """
        k = top_k or self.default_top_k
        depth = related_depth or self.max_graph_depth

        # Step 1: Standard vector search
        query_embedding = self.embedder.embed_query(query)
        primary_results = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=k,
        )

        # Step 2: Extract EIP references from query
        mentioned_eips = self._extract_eip_numbers(query)

        # Also extract EIPs from primary results (to understand context)
        for result in primary_results[:3]:  # Check top 3 results
            doc_eip = self._extract_eip_from_doc_id(result.chunk.document_id)
            if doc_eip:
                mentioned_eips.add(doc_eip)

        logger.debug("mentioned_eips", eips=mentioned_eips)

        # Step 3: Find related EIPs via graph
        related_eips: set[int] = set()
        dependency_context = {}

        if include_related and mentioned_eips:
            for eip_number in mentioned_eips:
                deps = self.traverser.get_dependencies(
                    eip_number,
                    direction=TraversalDirection.BOTH,
                    max_depth=depth,
                )
                related_eips.update(deps.get_related_eips())

                # Store context for response generation
                dependency_context[eip_number] = {
                    "direct_dependencies": deps.direct_dependencies,
                    "direct_dependents": deps.direct_dependents,
                    "supersedes": deps.supersedes,
                    "superseded_by": deps.superseded_by,
                }

            # Limit related EIPs
            if len(related_eips) > max_related_eips:
                # Prioritize: keep those that appear most frequently or are closest
                related_eips = self._prioritize_related_eips(
                    related_eips, mentioned_eips, max_related_eips
                )

        logger.debug("related_eips", count=len(related_eips), eips=related_eips)

        # Step 4: Fetch chunks from related EIPs
        related_results = []
        if related_eips:
            related_results = await self._fetch_related_chunks(
                query_embedding=query_embedding,
                related_eips=related_eips,
                per_eip_limit=self.related_top_k,
            )

        # Step 5: Combine and deduplicate results
        combined_results = self._combine_results(
            primary_results, related_results, max_total=k + len(related_eips) * 2
        )

        return GraphAugmentedResult(
            query=query,
            primary_results=primary_results,
            related_eips=related_eips,
            related_results=related_results,
            combined_results=combined_results,
            dependency_context=dependency_context,
        )

    def _extract_eip_numbers(self, text: str) -> set[int]:
        """Extract EIP numbers mentioned in text."""
        # Match patterns like "EIP-1559", "EIP 1559", "eip1559"
        matches = re.findall(r"eip[- ]?(\d+)", text, re.IGNORECASE)
        return set(int(m) for m in matches)

    def _extract_eip_from_doc_id(self, doc_id: str) -> int | None:
        """Extract EIP number from document ID."""
        # Assumes doc_id format like "eip-1559" or "EIP-1559"
        match = re.search(r"eip[- ]?(\d+)", doc_id, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _prioritize_related_eips(
        self,
        related_eips: set[int],
        mentioned_eips: set[int],
        limit: int,
    ) -> set[int]:
        """Prioritize which related EIPs to include.

        Prioritizes:
        1. Direct dependencies of mentioned EIPs
        2. Direct dependents of mentioned EIPs
        3. EIPs that supersede mentioned ones
        """
        prioritized = []

        for eip in mentioned_eips:
            # Add direct dependencies first
            deps = self.graph_store.get_direct_dependencies(eip)
            for dep in deps:
                if dep in related_eips and dep not in prioritized:
                    prioritized.append(dep)

            # Then direct dependents
            dependents = self.graph_store.get_direct_dependents(eip)
            for dependent in dependents:
                if dependent in related_eips and dependent not in prioritized:
                    prioritized.append(dependent)

            # Then supersedes
            supersedes = self.graph_store.get_supersedes(eip)
            superseded_by = self.graph_store.get_superseded_by(eip)
            for s in supersedes + superseded_by:
                if s in related_eips and s not in prioritized:
                    prioritized.append(s)

        # Add remaining up to limit
        for eip in related_eips:
            if eip not in prioritized:
                prioritized.append(eip)

        return set(prioritized[:limit])

    async def _fetch_related_chunks(
        self,
        query_embedding: list[float],
        related_eips: set[int],
        per_eip_limit: int,
    ) -> list[SearchResult]:
        """Fetch relevant chunks from related EIPs."""
        related_results = []

        for eip_number in related_eips:
            doc_id = f"eip-{eip_number}"
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=per_eip_limit,
                document_filter=doc_id,
            )
            related_results.extend(results)

        return related_results

    def _combine_results(
        self,
        primary: list[SearchResult],
        related: list[SearchResult],
        max_total: int,
    ) -> list[SearchResult]:
        """Combine and deduplicate primary and related results."""
        seen_chunk_ids = set()
        combined = []

        # Add primary results first (higher priority)
        for result in primary:
            if result.chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(result.chunk.chunk_id)
                combined.append(result)

        # Add related results
        for result in related:
            if result.chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(result.chunk.chunk_id)
                combined.append(result)

        # Sort by similarity and limit
        combined.sort(key=lambda r: r.similarity, reverse=True)
        return combined[:max_total]

    def format_context_with_graph(
        self,
        result: GraphAugmentedResult,
        max_tokens: int | None = None,
    ) -> str:
        """Format retrieval results with graph context included.

        Adds a preamble explaining relationships between EIPs.
        """
        parts = []

        # Add relationship context if available
        if result.dependency_context:
            rel_summary = self._format_relationship_summary(result.dependency_context)
            if rel_summary:
                parts.append(f"## EIP Relationships\n{rel_summary}\n")

        # Add retrieved chunks
        token_count = sum(len(p.split()) for p in parts)  # Rough estimate
        for r in result.combined_results:
            chunk = r.chunk
            chunk_tokens = chunk.token_count

            if max_tokens and token_count + chunk_tokens > max_tokens:
                break

            source = chunk.document_id.upper()
            if chunk.section_path:
                source = f"{source} ({chunk.section_path})"

            parts.append(f"[Source: {source}]\n{chunk.content}\n")
            token_count += chunk_tokens

        return "\n---\n".join(parts)

    def _format_relationship_summary(
        self,
        dependency_context: dict,
    ) -> str:
        """Format a summary of EIP relationships."""
        lines = []

        for eip_number, context in dependency_context.items():
            deps = context.get("direct_dependencies", [])
            dependents = context.get("direct_dependents", [])
            supersedes = context.get("supersedes", [])
            superseded_by = context.get("superseded_by", [])

            if deps:
                dep_str = ", ".join(f"EIP-{d}" for d in deps)
                lines.append(f"- EIP-{eip_number} requires: {dep_str}")

            if dependents:
                dep_str = ", ".join(f"EIP-{d}" for d in dependents[:5])
                suffix = f" (and {len(dependents) - 5} more)" if len(dependents) > 5 else ""
                lines.append(f"- EIP-{eip_number} is required by: {dep_str}{suffix}")

            if supersedes:
                sup_str = ", ".join(f"EIP-{s}" for s in supersedes)
                lines.append(f"- EIP-{eip_number} supersedes: {sup_str}")

            if superseded_by:
                sup_str = ", ".join(f"EIP-{s}" for s in superseded_by)
                lines.append(f"- EIP-{eip_number} is superseded by: {sup_str}")

        return "\n".join(lines)


async def get_eip_context_for_query(
    query: str,
    graph_store: FalkorDBStore,
    max_depth: int = 2,
) -> dict:
    """Get graph context relevant to a query.

    Standalone function for when you need just the graph context
    without doing retrieval.
    """
    traverser = DependencyTraverser(graph_store, max_depth=max_depth)

    # Extract EIP numbers from query
    matches = re.findall(r"eip[- ]?(\d+)", query, re.IGNORECASE)
    mentioned_eips = set(int(m) for m in matches)

    context = {}
    for eip_number in mentioned_eips:
        deps = traverser.get_dependencies(
            eip_number,
            direction=TraversalDirection.BOTH,
        )
        context[eip_number] = {
            "exists_in_graph": graph_store.get_eip(eip_number) is not None,
            "direct_dependencies": deps.direct_dependencies,
            "direct_dependents": deps.direct_dependents,
            "all_dependencies": [d.eip_number for d in deps.all_dependencies],
            "all_dependents": [d.eip_number for d in deps.all_dependents],
            "supersedes": deps.supersedes,
            "superseded_by": deps.superseded_by,
        }

    return context
