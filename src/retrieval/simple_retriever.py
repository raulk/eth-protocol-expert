"""Simple Retriever - Basic vector search retrieval."""

import asyncio
from dataclasses import dataclass

import structlog

from ..embeddings.voyage_embedder import VoyageEmbedder
from ..storage.pg_vector_store import PgVectorStore, SearchResult

logger = structlog.get_logger()


@dataclass
class RetrievalResult:
    """Result from retrieval with context."""
    query: str
    results: list[SearchResult]
    total_tokens: int


class SimpleRetriever:
    """Simple top-k vector retrieval (Phase 0).

    Takes a query, embeds it, and returns the most similar chunks.
    """

    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: PgVectorStore,
        default_top_k: int = 10,
    ):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_filter: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve (default: 10)
            document_filter: Optional document_id to filter by

        Returns:
            RetrievalResult with ranked chunks
        """
        k = top_k or self.default_top_k

        # Embed the query
        query_embedding = await asyncio.to_thread(self.embedder.embed_query, query)

        # Search for similar chunks
        results = await self.store.search(
            query_embedding=query_embedding,
            limit=k,
            document_filter=document_filter,
        )

        # Calculate total tokens
        total_tokens = sum(r.chunk.token_count for r in results)

        logger.debug(
            "retrieved_chunks",
            query=query[:50],
            num_results=len(results),
            total_tokens=total_tokens,
            top_similarity=results[0].similarity if results else 0,
        )

        return RetrievalResult(
            query=query,
            results=results,
            total_tokens=total_tokens,
        )

    def format_context(
        self,
        results: list[SearchResult],
        max_tokens: int | None = None,
    ) -> str:
        """Format retrieved chunks into context string.

        Args:
            results: Search results to format
            max_tokens: Optional token limit

        Returns:
            Formatted context string
        """
        context_parts = []
        token_count = 0

        for result in results:
            chunk = result.chunk

            # Check token limit
            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            # Format the chunk with source info
            source = chunk.document_id.upper()
            if chunk.section_path:
                source = f"{source} ({chunk.section_path})"

            context_parts.append(
                f"[Source: {source}]\n{chunk.content}\n"
            )
            token_count += chunk.token_count

        return "\n---\n".join(context_parts)

    def format_context_with_citations(
        self,
        results: list[SearchResult],
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict]]:
        """Format context with citation metadata (Phase 1).

        Returns:
            Tuple of (context_string, citation_list)
        """
        context_parts = []
        citations = []
        token_count = 0

        for i, result in enumerate(results):
            chunk = result.chunk
            citation_id = f"[{i + 1}]"

            # Check token limit
            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            # Build citation info
            citation = {
                "id": citation_id,
                "document_id": chunk.document_id,
                "section": chunk.section_path or "Full document",
                "chunk_id": chunk.chunk_id,
                "similarity": result.similarity,
            }
            citations.append(citation)

            # Format chunk with citation marker
            context_parts.append(
                f"{citation_id} [{chunk.document_id.upper()}"
                f"{' - ' + chunk.section_path if chunk.section_path else ''}]\n"
                f"{chunk.content}"
            )
            token_count += chunk.token_count

        context = "\n\n---\n\n".join(context_parts)
        return context, citations
