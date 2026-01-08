"""Reranker - Rerank search results using Cohere rerank-v3."""

import os
from dataclasses import dataclass

import cohere
import structlog

from ..storage.pg_vector_store import StoredChunk
from .hybrid_retriever import HybridResult

logger = structlog.get_logger()


@dataclass
class RerankResult:
    """A reranked search result."""

    chunk: StoredChunk
    relevance_score: float
    original_index: int


class CohereReranker:
    """Rerank search results using Cohere's rerank-english-v3.0 model.

    Reranking improves retrieval quality by using a more sophisticated
    cross-encoder model to score query-document pairs, rather than relying
    solely on embedding similarity or BM25 scores.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Cohere rerank model (default: rerank-english-v3.0)
        """
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = cohere.Client(self.api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> list[RerankResult]:
        """Rerank hybrid search results using Cohere.

        Args:
            query: The search query
            results: List of HybridResult from hybrid retrieval
            top_n: Number of top results to return (default: all)
            return_documents: Whether to include document text in API response

        Returns:
            List of RerankResult sorted by relevance score (highest first)
        """
        if not results:
            return []

        # Extract document texts for reranking
        documents = [r.chunk.content for r in results]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_n or len(results),
            return_documents=return_documents,
        )

        reranked = []
        for result in response.results:
            original_idx = result.index
            original_result = results[original_idx]

            reranked.append(
                RerankResult(
                    chunk=original_result.chunk,
                    relevance_score=result.relevance_score,
                    original_index=original_idx,
                )
            )

        logger.debug(
            "cohere_rerank",
            query=query[:50],
            input_count=len(results),
            output_count=len(reranked),
            top_score=reranked[0].relevance_score if reranked else 0,
        )

        return reranked

    def rerank_chunks(
        self,
        query: str,
        chunks: list[StoredChunk],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank a list of chunks directly.

        Convenience method for reranking chunks without HybridResult wrapper.

        Args:
            query: The search query
            chunks: List of StoredChunk to rerank
            top_n: Number of top results to return

        Returns:
            List of RerankResult sorted by relevance
        """
        if not chunks:
            return []

        documents = [c.content for c in chunks]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_n or len(chunks),
        )

        reranked = []
        for result in response.results:
            original_idx = result.index
            original_chunk = chunks[original_idx]

            reranked.append(
                RerankResult(
                    chunk=original_chunk,
                    relevance_score=result.relevance_score,
                    original_index=original_idx,
                )
            )

        return reranked

    def format_context(
        self,
        results: list[RerankResult],
        max_tokens: int | None = None,
    ) -> str:
        """Format reranked results into context string."""
        context_parts = []
        token_count = 0

        for result in results:
            chunk = result.chunk

            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            source = chunk.document_id.upper()
            if chunk.section_path:
                source = f"{source} ({chunk.section_path})"

            context_parts.append(f"[Source: {source}]\n{chunk.content}\n")
            token_count += chunk.token_count

        return "\n---\n".join(context_parts)

    def format_context_with_citations(
        self,
        results: list[RerankResult],
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict]]:
        """Format reranked results with citation metadata.

        Returns:
            Tuple of (context_string, citation_list)
        """
        context_parts = []
        citations = []
        token_count = 0

        for i, result in enumerate(results):
            chunk = result.chunk
            citation_id = f"[{i + 1}]"

            if max_tokens and token_count + chunk.token_count > max_tokens:
                break

            citation = {
                "id": citation_id,
                "document_id": chunk.document_id,
                "section": chunk.section_path or "Full document",
                "chunk_id": chunk.chunk_id,
                "relevance_score": result.relevance_score,
                "original_rank": result.original_index + 1,
            }
            citations.append(citation)

            context_parts.append(
                f"{citation_id} [{chunk.document_id.upper()}"
                f"{' - ' + chunk.section_path if chunk.section_path else ''}]\n"
                f"{chunk.content}"
            )
            token_count += chunk.token_count

        context = "\n\n---\n\n".join(context_parts)
        return context, citations


class RerankedHybridRetriever:
    """Convenience class combining HybridRetriever with CohereReranker.

    Provides a single interface for hybrid search + reranking pipeline.
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker: CohereReranker,
    ):
        """Initialize reranked hybrid retriever.

        Args:
            hybrid_retriever: HybridRetriever instance
            reranker: CohereReranker instance
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        initial_limit: int | None = None,
        metadata_query=None,
    ) -> list[RerankResult]:
        """Retrieve and rerank documents.

        Args:
            query: Natural language query
            limit: Number of final results after reranking
            initial_limit: Number of results to fetch before reranking (default: 3x limit)
            metadata_query: Optional metadata filters

        Returns:
            List of RerankResult sorted by relevance
        """
        fetch_limit = initial_limit or (limit * 3)

        # Get hybrid results
        hybrid_result = await self.hybrid_retriever.retrieve(
            query=query,
            limit=fetch_limit,
            metadata_query=metadata_query,
        )

        # Rerank
        reranked = self.reranker.rerank(
            query=query,
            results=hybrid_result.results,
            top_n=limit,
        )

        logger.info(
            "reranked_hybrid_retrieval",
            query=query[:50],
            hybrid_count=len(hybrid_result.results),
            reranked_count=len(reranked),
        )

        return reranked
